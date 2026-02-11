#!/usr/bin/env python3
"""
RAFT Dense Motion Stabilization for Video Stabbot

Uses RAFT (Recurrent All-Pairs Field Transforms) deep learning model
for dense optical flow estimation and trajectory-based stabilization.

Requires: torch, torchvision, opencv-python, numpy, scipy

Model: RAFT-sintel or RAFT-things (auto-downloaded on first run)
"""

import argparse
import json
import sys
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Import smoothing library
sys.path.insert(0, os.path.dirname(__file__))
from smoothing_lib import apply_smoothing


def emit(phase, progress, **extra):
    """Emit progress as JSON to stdout"""
    msg = {"phase": phase, "progress": round(progress, 2), **extra}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def emit_error(message):
    """Emit error message to stderr"""
    sys.stderr.write(json.dumps({"error": message}) + "\n")
    sys.stderr.flush()


def load_raft_model(model_name='raft-sintel'):
    """
    Load RAFT model from torchvision or local checkpoint

    Args:
        model_name: 'raft-sintel' or 'raft-things'

    Returns:
        model: RAFT model in eval mode
        device: torch device
    """
    try:
        import torch
        import torchvision.models.optical_flow as flow_models
    except ImportError:
        emit_error(
            "PyTorch not installed.\n\n"
            "RAFT mode requires PyTorch and torchvision.\n"
            "Install with: pip install torch torchvision\n\n"
            "Warning: This is a large download (~2-4GB)."
        )
        return None, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emit('loading', 5, message=f"Loading RAFT model on {device}")

    try:
        if model_name == 'raft-sintel':
            # RAFT trained on Sintel dataset (better for natural videos)
            weights = flow_models.Raft_Large_Weights.C_T_SKHT_V2
            model = flow_models.raft_large(weights=weights)
        else:
            # RAFT trained on Things dataset (more general)
            weights = flow_models.Raft_Large_Weights.DEFAULT
            model = flow_models.raft_large(weights=weights)

        model = model.to(device).eval()
        emit('loading', 15, message="RAFT model loaded")
        return model, device

    except Exception as e:
        emit_error(f"Failed to load RAFT model: {str(e)}")
        return None, None


def preprocess_frame(frame, device):
    """
    Preprocess frame for RAFT model

    Args:
        frame: OpenCV BGR frame
        device: torch device

    Returns:
        tensor: Preprocessed frame tensor [1, 3, H, W]
    """
    import torch

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to float [0, 1] and then to tensor
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()

    # Normalize to [0, 255] (RAFT expects this range)
    frame_tensor = frame_tensor.unsqueeze(0).to(device)

    return frame_tensor


def estimate_flow(model, device, frame1, frame2, max_iterations=12):
    """
    Estimate optical flow between two frames using RAFT

    Args:
        model: RAFT model
        device: torch device
        frame1: First frame (OpenCV BGR)
        frame2: Second frame (OpenCV BGR)
        max_iterations: Number of RAFT iterations (default: 12)

    Returns:
        flow: Optical flow array [H, W, 2]
    """
    import torch

    # Preprocess frames
    img1 = preprocess_frame(frame1, device)
    img2 = preprocess_frame(frame2, device)

    # Estimate flow
    with torch.no_grad():
        # RAFT returns a list of flow predictions (refinements)
        flow_predictions = model(img1, img2, num_flow_updates=max_iterations)

        # Take the final (most refined) prediction
        flow = flow_predictions[-1]

    # Convert to numpy [H, W, 2]
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()

    return flow_np


def flow_to_transform(flow):
    """
    Convert dense optical flow to affine transform parameters

    Uses median flow vectors as a robust estimate of camera motion

    Args:
        flow: Dense optical flow [H, W, 2]

    Returns:
        transform: [dx, dy, da] - translation x, y, and rotation angle
    """
    # Use median flow as robust camera motion estimate
    # (median is less affected by moving objects)

    dx = np.median(flow[:, :, 0])
    dy = np.median(flow[:, :, 1])

    # Estimate rotation from flow field using corner points
    # (corners are more informative for rotation)
    h, w = flow.shape[:2]

    # Sample corner regions
    corner_size = min(h, w) // 8
    corners = [
        flow[:corner_size, :corner_size],  # top-left
        flow[:corner_size, -corner_size:],  # top-right
        flow[-corner_size:, :corner_size],  # bottom-left
        flow[-corner_size:, -corner_size:],  # bottom-right
    ]

    # Compute rotation from corner flow differences
    # Simplified approach: use median flow differences
    angles = []
    for corner in corners:
        if corner.size > 0:
            flow_x = np.median(corner[:, :, 0])
            flow_y = np.median(corner[:, :, 1])
            angle = np.arctan2(flow_y, flow_x)
            angles.append(angle)

    da = np.median(angles) if angles else 0.0

    # Convert to small angle (rotation is usually small in video)
    da = np.clip(da, -0.1, 0.1)  # Limit to ~5 degrees

    return np.array([dx, dy, da])


def process_video(args):
    """
    Main processing pipeline for RAFT stabilization

    Three-pass approach:
    1. Flow estimation and trajectory extraction
    2. Trajectory smoothing
    3. Frame warping and encoding
    """
    # Load RAFT model
    model, device = load_raft_model(args.raft_model)
    if model is None:
        return None

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        emit_error(f"Cannot open video: {args.input}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    emit('flow', 20, message="Starting flow estimation")

    # ============================================================
    # Pass 1: Flow Estimation (20-50%)
    # ============================================================

    trajectory = []
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            # Estimate flow between consecutive frames
            flow = estimate_flow(model, device, prev_frame, frame, args.max_iterations)

            # Convert flow to transform
            transform = flow_to_transform(flow)
            trajectory.append(transform)

            # Update progress
            progress = 20 + (frame_idx / total_frames) * 30
            if frame_idx % 10 == 0:
                emit('flow', progress, message=f"Flow estimation ({frame_idx}/{total_frames})")

        prev_frame = frame.copy()
        frame_idx += 1

    cap.release()

    if len(trajectory) == 0:
        emit_error("No frames processed")
        return None

    emit('trajectory', 50, message="Smoothing trajectory")

    # ============================================================
    # Pass 2: Trajectory Smoothing (50-60%)
    # ============================================================

    trajectory = np.array(trajectory)

    # Apply smoothing to each component
    smoothed = apply_smoothing(
        trajectory,
        method=args.smoothing_method,
        window=args.smoothing_strength,
        sigma=args.smoothing_strength / 2.0,
        smoothing_factor=args.smoothing_strength * 100,
    )

    # Compute stabilization transforms (difference between original and smoothed)
    stabilization_transforms = trajectory - smoothed

    emit('transform', 60, message="Applying stabilization")

    # ============================================================
    # Pass 3: Frame Warping and Encoding (60-100%)
    # ============================================================

    # Reopen video for second pass
    cap = cv2.VideoCapture(args.input)

    # Create temporary output using OpenCV
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_fd)

    # Output resolution
    if args.resolution != 'source':
        if args.resolution == '1080p':
            out_width, out_height = 1920, 1080
        elif args.resolution == '720p':
            out_width, out_height = 1280, 720
        elif args.resolution == '480p':
            out_width, out_height = 854, 480
        else:
            out_width, out_height = width, height
    else:
        out_width, out_height = width, height

    # Crop percentage
    crop_factor = 1.0 - (args.crop_percent / 100.0)
    crop_width = int(out_width * crop_factor)
    crop_height = int(out_height * crop_factor)

    # FFmpeg writer
    ffmpeg_cmd = [
        args.ffmpeg,
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{crop_width}x{crop_height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        temp_path
    ]

    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    frame_idx = 0
    cumulative_transform = np.zeros(3)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get stabilization transform for this frame
        if frame_idx < len(stabilization_transforms):
            delta = stabilization_transforms[frame_idx]
            cumulative_transform += delta

        # Build affine transformation matrix
        dx, dy, da = cumulative_transform

        # Rotation matrix
        M_rot = cv2.getRotationMatrix2D((width / 2, height / 2), np.degrees(da), 1.0)

        # Add translation
        M_rot[0, 2] += dx
        M_rot[1, 2] += dy

        # Warp frame
        stabilized = cv2.warpAffine(frame, M_rot, (width, height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

        # Crop to remove borders
        crop_x = (width - crop_width) // 2
        crop_y = (height - crop_height) // 2
        cropped = stabilized[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        # Resize to output resolution if needed
        if (crop_width, crop_height) != (out_width, out_height):
            cropped = cv2.resize(cropped, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

        # Write to FFmpeg
        ffmpeg_process.stdin.write(cropped.tobytes())

        # Update progress
        progress = 60 + (frame_idx / total_frames) * 40
        if frame_idx % 10 == 0:
            emit('transform', progress, message=f"Stabilizing ({frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    if ffmpeg_process.returncode != 0:
        emit_error("FFmpeg encoding failed")
        return None

    # Move temp file to final output
    import shutil
    shutil.move(temp_path, args.output)

    emit('transform', 100, message="Stabilization complete")

    # Get output size
    output_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0

    return {"ok": True, "outputSize": output_size}


def main():
    parser = argparse.ArgumentParser(description="RAFT dense motion stabilization")

    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output video file')
    parser.add_argument('--ffmpeg', required=True, help='Path to FFmpeg')
    parser.add_argument('--raft-model', default='raft-sintel',
                       choices=['raft-sintel', 'raft-things'],
                       help='RAFT model variant (sintel for natural videos)')
    parser.add_argument('--max-iterations', type=int, default=12,
                       help='RAFT refinement iterations (6-20)')
    parser.add_argument('--smoothing-method', default='savitzky_golay',
                       choices=['moving_average', 'savitzky_golay', 'gaussian', 'spline'],
                       help='Trajectory smoothing method')
    parser.add_argument('--smoothing-strength', type=int, default=30,
                       help='Smoothing window/strength (10-100)')
    parser.add_argument('--crop-percent', type=int, default=10,
                       help='Border crop percentage (0-30)')
    parser.add_argument('--resolution', default='source',
                       choices=['source', '1080p', '720p', '480p'],
                       help='Output resolution')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        emit_error(f"Input file not found: {args.input}")
        return 1

    if not os.path.exists(args.ffmpeg):
        emit_error(f"FFmpeg not found: {args.ffmpeg}")
        return 1

    # Run processing
    result = process_video(args)

    if result is None:
        return 1

    # Emit final result
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        emit_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
