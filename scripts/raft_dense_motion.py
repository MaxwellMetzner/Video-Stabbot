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
    emit('loading', 1, message=f"Loading RAFT model on {device}")

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
        emit('loading', 4, message="RAFT model loaded")
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
        tensor: Preprocessed frame tensor [1, 3, H, W] in [-1, 1]
    """
    import torch

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to float tensor [1, 3, H, W]
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0)

    # Normalize from [0, 255] to [-1, 1]  (required by torchvision RAFT)
    frame_tensor = frame_tensor / 255.0            # [0, 1]
    frame_tensor = (frame_tensor - 0.5) / 0.5      # [-1, 1]

    return frame_tensor.to(device)


def pad_to_divisible(tensor, divisor=8):
    """
    Pad a [1, C, H, W] tensor so H and W are divisible by `divisor`.
    Returns (padded_tensor, (pad_h, pad_w)).
    """
    import torch.nn.functional as F
    _, _, h, w = tensor.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')
    return tensor, (pad_h, pad_w)


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

    # RAFT requires dimensions divisible by 8 — pad if necessary
    orig_h, orig_w = img1.shape[2], img1.shape[3]
    img1, (pad_h, pad_w) = pad_to_divisible(img1, 8)
    img2, _ = pad_to_divisible(img2, 8)

    # Estimate flow
    with torch.no_grad():
        # RAFT returns a list of flow predictions (refinements)
        flow_predictions = model(img1, img2, num_flow_updates=max_iterations)

        # Take the final (most refined) prediction
        flow = flow_predictions[-1]

    # Crop padding back to original dimensions
    flow = flow[:, :, :orig_h, :orig_w]

    # Convert to numpy [H, W, 2]
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()

    return flow_np


def flow_to_transform(flow):
    """
    Convert dense optical flow to affine transform [dx, dy, da].

    Builds point correspondences from the flow field and uses
    cv2.estimateAffinePartial2D with RANSAC — the exact same method
    the working OpenCV feature-tracking script uses.

    Args:
        flow: Dense optical flow [H, W, 2]

    Returns:
        transform: [dx, dy, da] - translation x, y, and rotation angle
    """
    h, w = flow.shape[:2]

    # Subsample the flow field into point correspondences
    step = max(1, min(h, w) // 80)
    ys, xs = np.mgrid[0:h:step, 0:w:step]

    pts1 = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    flow_sub = flow[0:h:step, 0:w:step]
    pts2 = pts1 + flow_sub.reshape(-1, 2).astype(np.float32)

    # Estimate similarity transform (rotation + uniform scale + translation)
    # RANSAC rejects outliers from moving objects and unreliable flow
    M, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )

    if M is not None:
        dx = float(M[0, 2])
        dy = float(M[1, 2])
        da = float(np.arctan2(M[1, 0], M[0, 0]))
    else:
        dx, dy, da = 0.0, 0.0, 0.0

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

    emit('flow', 5, message="Starting flow estimation")

    # ============================================================
    # Pass 1: Flow Estimation (5-90%)
    # ============================================================

    transforms = [np.array([0.0, 0.0, 0.0])]  # Frame 0: no previous → zero motion
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            # Estimate flow between consecutive frames
            flow = estimate_flow(model, device, prev_frame, frame, args.max_iterations)

            # Convert flow to transform [dx, dy, da]
            transform = flow_to_transform(flow)
            transforms.append(transform)

            # Update progress
            progress = 5 + (frame_idx / total_frames) * 85
            if frame_idx % 10 == 0:
                emit('flow', progress, message=f"Flow estimation ({frame_idx}/{total_frames})")

        prev_frame = frame.copy()
        frame_idx += 1

    cap.release()

    if len(transforms) <= 1:
        emit_error("No frames processed")
        return None

    # ============================================================
    # Pass 2: Trajectory Smoothing (91-92%)
    # ============================================================

    transforms = np.array(transforms)

    emit('trajectory', 91, message="Smoothing trajectory")

    # Build cumulative trajectory (same approach as OpenCV script)
    # trajectory[0] = [0,0,0] (frame 0 is the origin)
    cumulative_trajectory = np.cumsum(transforms, axis=0)

    # Convert smoothing strength to method-specific parameters
    if args.smoothing_method == 'moving_average':
        smooth_params = {'window': int(args.smoothing_strength)}
    elif args.smoothing_method == 'savgol':
        window = int(args.smoothing_strength)
        if window % 2 == 0:
            window += 1
        smooth_params = {'window': window, 'polyorder': 3}
    elif args.smoothing_method == 'gaussian':
        smooth_params = {'sigma': args.smoothing_strength / 10.0}
    elif args.smoothing_method == 'spline':
        smooth_params = {'smoothing_factor': args.smoothing_strength}
    else:
        smooth_params = {'window': int(args.smoothing_strength)}

    # Smooth the cumulative trajectory
    smooth_trajectory = apply_smoothing(
        cumulative_trajectory,
        method=args.smoothing_method,
        **smooth_params,
    )

    # Correction = smoothed path - original path (applied per-frame)
    correction = smooth_trajectory - cumulative_trajectory

    max_corr = np.max(np.abs(correction), axis=0)
    emit('trajectory', 92, message=(
        f"Trajectory smoothed — max correction: dx={max_corr[0]:.1f} dy={max_corr[1]:.1f} da={max_corr[2]:.4f}"
    ))

    emit('transform', 92, message="Applying stabilization")

    # ============================================================
    # Pass 3: Frame Warping and Encoding (92-100%)
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

    # Crop percentage (applied after resizing to output resolution)
    crop_percent = args.crop_percent / 100.0
    crop_width = int(out_width * (1 - crop_percent))
    crop_height = int(out_height * (1 - crop_percent))
    crop_x = (out_width - crop_width) // 2
    crop_y = (out_height - crop_height) // 2

    # FFmpeg writer — dimensions must match the final cropped frame
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get correction for this frame (already computed per-frame)
        if frame_idx < len(correction):
            dx, dy, da = correction[frame_idx]
        else:
            dx, dy, da = 0.0, 0.0, 0.0

        # Build affine transform matrix (same as OpenCV script)
        transform_matrix = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da),  np.cos(da), dy]
        ], dtype=np.float32)

        # Warp frame
        stabilized = cv2.warpAffine(frame, transform_matrix, (width, height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

        # Resize to output resolution first (if needed)
        if (out_width, out_height) != (width, height):
            stabilized = cv2.resize(stabilized, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

        # Then crop to remove black borders
        cropped = stabilized[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        # Write to FFmpeg
        ffmpeg_process.stdin.write(cropped.tobytes())

        # Update progress
        progress = 92 + (frame_idx / total_frames) * 8
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
    parser.add_argument('--smoothing-method', default='savgol',
                       choices=['moving_average', 'savgol', 'gaussian', 'spline'],
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
