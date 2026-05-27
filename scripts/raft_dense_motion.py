#!/usr/bin/env python3
"""
RAFT dense-motion video stabilization.

RAFT estimates dense optical flow. This script turns that flow into a robust
global camera transform, smooths the resulting trajectory, warps frames, and
muxes the source audio back into the stabilized output.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from smoothing_lib import apply_smoothing


MIN_TRACKS = 12
DEFAULT_MESH_GRID = "7x9"


def emit(phase, progress, **extra):
    """Emit progress as JSON to stdout."""
    msg = {"phase": phase, "progress": round(progress, 2), **extra}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def emit_error(message):
    """Emit error message to stderr."""
    sys.stderr.write(json.dumps({"error": message}) + "\n")
    sys.stderr.flush()


def load_raft_model(model_name="raft-sintel"):
    try:
        import torch
        import torchvision.models.optical_flow as flow_models
    except ImportError:
        emit_error(
            "PyTorch or torchvision is not installed.\n\n"
            "RAFT mode requires torch and torchvision.\n"
            "Install with: pip install torch torchvision"
        )
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit("loading", 1, message=f"Loading RAFT model on {device}")

    try:
        weights_enum = flow_models.Raft_Large_Weights
        if model_name == "raft-things":
            weights = getattr(weights_enum, "C_T_V2", getattr(weights_enum, "C_T_V1", weights_enum.DEFAULT))
        else:
            weights = getattr(weights_enum, "C_T_SKHT_V2", weights_enum.DEFAULT)

        model = flow_models.raft_large(weights=weights, progress=False).to(device).eval()
        emit("loading", 4, message="RAFT model loaded")
        return model, device
    except Exception as exc:
        emit_error(f"Failed to load RAFT model: {exc}")
        return None, None


def analysis_frames(frame1, frame2, max_edge):
    height, width = frame1.shape[:2]
    scale = min(1.0, max_edge / float(max(height, width)))

    new_width = max(128, int(round(width * scale / 8.0)) * 8)
    new_height = max(128, int(round(height * scale / 8.0)) * 8)
    new_width = max(8, new_width)
    new_height = max(8, new_height)

    if (new_width, new_height) == (width, height):
        return frame1, frame2, 1.0, 1.0

    resized1 = cv2.resize(frame1, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(frame2, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized1, resized2, width / float(new_width), height / float(new_height)


def preprocess_frame(frame, device):
    import torch

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
    tensor = tensor / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.to(device)


def pad_to_divisible(tensor, divisor=8):
    import torch.nn.functional as functional

    _, _, height, width = tensor.shape
    pad_height = (divisor - height % divisor) % divisor
    pad_width = (divisor - width % divisor) % divisor
    if pad_height or pad_width:
        tensor = functional.pad(tensor, (0, pad_width, 0, pad_height), mode="replicate")
    return tensor, pad_height, pad_width


def estimate_flow(model, device, frame1, frame2, max_iterations, max_edge):
    import torch

    resized1, resized2, scale_x, scale_y = analysis_frames(frame1, frame2, max_edge)
    img1 = preprocess_frame(resized1, device)
    img2 = preprocess_frame(resized2, device)

    orig_height, orig_width = img1.shape[2], img1.shape[3]
    img1, _pad_h, _pad_w = pad_to_divisible(img1, 8)
    img2, _pad_h2, _pad_w2 = pad_to_divisible(img2, 8)

    with torch.inference_mode():
        flow_predictions = model(img1, img2, num_flow_updates=max_iterations)
        flow = flow_predictions[-1][:, :, :orig_height, :orig_width]

    flow_np = flow[0].permute(1, 2, 0).detach().cpu().numpy()
    return flow_np, resized1, resized2, scale_x, scale_y


def decompose_affine(matrix, scale_x=1.0, scale_y=1.0):
    if matrix is None:
        return np.zeros(4, dtype=np.float64)

    dx = float(matrix[0, 2]) * scale_x
    dy = float(matrix[1, 2]) * scale_y
    angle = float(math.atan2(matrix[1, 0], matrix[0, 0]))
    scale = float(math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    return np.array([dx, dy, angle, scale - 1.0], dtype=np.float64)


def valid_motion(motion, width, height, inliers, total):
    if total < MIN_TRACKS or inliers < MIN_TRACKS:
        return False
    if inliers / max(total, 1) < 0.30:
        return False

    max_shift = max(width, height) * 0.35
    if abs(motion[0]) > max_shift or abs(motion[1]) > max_shift:
        return False
    if abs(motion[2]) > math.radians(25):
        return False
    if abs(motion[3]) > 0.20:
        return False
    return True


def parse_mesh_grid(value):
    if not value or str(value).lower() == "off":
        return None

    try:
        rows_text, cols_text = str(value).lower().split("x", 1)
        rows = int(rows_text)
        cols = int(cols_text)
    except ValueError:
        rows, cols = 7, 9

    return max(3, rows), max(3, cols)


def affine_flow_field(matrix, width, height):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    pred_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2] - xx
    pred_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2] - yy
    return np.dstack([pred_x, pred_y]).astype(np.float32)


def residual_inlier_mask(flow, matrix):
    height, width = flow.shape[:2]
    if matrix is None:
        return np.isfinite(flow).all(axis=2), None

    predicted = affine_flow_field(matrix, width, height)
    residual = np.linalg.norm(flow - predicted, axis=2)
    finite = np.isfinite(residual)
    if not np.any(finite):
        return finite, predicted

    finite_residual = residual[finite]
    median = float(np.median(finite_residual))
    mad = float(np.median(np.abs(finite_residual - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(2.0, median + 2.5 * robust_sigma)
    threshold = min(threshold, float(np.percentile(finite_residual, 80)))
    threshold = max(threshold, 2.0)
    return finite & (residual <= threshold), predicted


def mesh_motion_from_flow(flow, matrix, scale_x, scale_y, rows, cols):
    height, width = flow.shape[:2]
    inlier_mask, predicted = residual_inlier_mask(flow, matrix)
    if predicted is None:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        predicted = affine_flow_field(identity, width, height)

    xs = np.linspace(0, width - 1, cols)
    ys = np.linspace(0, height - 1, rows)
    radius_x = max(8, int(round(width / max(cols - 1, 1) * 0.75)))
    radius_y = max(8, int(round(height / max(rows - 1, 1) * 0.75)))

    mesh = np.zeros((rows, cols, 2), dtype=np.float64)

    for row, y in enumerate(ys):
        y0 = max(0, int(round(y - radius_y)))
        y1 = min(height, int(round(y + radius_y + 1)))
        for col, x in enumerate(xs):
            x0 = max(0, int(round(x - radius_x)))
            x1 = min(width, int(round(x + radius_x + 1)))
            local_flow = flow[y0:y1, x0:x1]
            local_mask = inlier_mask[y0:y1, x0:x1]

            if np.count_nonzero(local_mask) >= MIN_TRACKS:
                vector = np.median(local_flow[local_mask], axis=0)
            else:
                px = int(np.clip(round(x), 0, width - 1))
                py = int(np.clip(round(y), 0, height - 1))
                vector = predicted[py, px]

            mesh[row, col, 0] = float(vector[0]) * scale_x
            mesh[row, col, 1] = float(vector[1]) * scale_y

    for channel in range(2):
        mesh[:, :, channel] = cv2.GaussianBlur(mesh[:, :, channel], (3, 3), 0)

    return mesh


def dense_flow_to_motion(flow, scale_x, scale_y):
    height, width = flow.shape[:2]
    step = max(4, min(height, width) // 80)
    ys, xs = np.mgrid[0:height:step, 0:width:step]

    pts1 = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    flow_sub = flow[0:height:step, 0:width:step].reshape(-1, 2).astype(np.float32)
    finite = np.isfinite(flow_sub).all(axis=1)
    pts1 = pts1[finite]
    flow_sub = flow_sub[finite]

    if len(pts1) < MIN_TRACKS:
        return None, 0, 0, None

    magnitudes = np.linalg.norm(flow_sub, axis=1)
    if len(magnitudes) >= 50:
        limit = np.percentile(magnitudes, 90)
        keep = magnitudes <= max(limit, 1.0)
        if np.count_nonzero(keep) >= MIN_TRACKS:
            pts1 = pts1[keep]
            flow_sub = flow_sub[keep]

    pts2 = pts1 + flow_sub
    matrix, inliers = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        return None, 0, len(pts1), None

    motion = decompose_affine(matrix, scale_x, scale_y)
    return motion, int(np.count_nonzero(inliers)), len(pts1), matrix


def sparse_fallback_motion(prev_frame, curr_frame, scale_x, scale_y):
    prev_gray = cv2.equalizeHist(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    curr_gray = cv2.equalizeHist(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY))
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1500,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if prev_pts is None or len(prev_pts) < MIN_TRACKS:
        return None, 0, 0

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=1e-4,
    )
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    if curr_pts is None or status is None:
        return None, 0, len(prev_pts)

    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, curr_pts, None, **lk_params)
    if back_pts is None or back_status is None:
        return None, 0, 0

    fb_error = np.linalg.norm(prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
    valid = (status.ravel() == 1) & (back_status.ravel() == 1) & (fb_error < 1.5)
    if np.count_nonzero(valid) < MIN_TRACKS:
        return None, 0, len(prev_pts)

    good_prev = prev_pts.reshape(-1, 2)[valid]
    good_curr = curr_pts.reshape(-1, 2)[valid]
    matrix, inliers = cv2.estimateAffinePartial2D(
        good_prev,
        good_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        return None, 0, len(good_prev)

    return decompose_affine(matrix, scale_x, scale_y), int(np.count_nonzero(inliers)), len(good_prev)


def clamp_transforms(transforms, width, height):
    clamped = np.array(transforms, dtype=np.float64, copy=True)
    max_shift = max(width, height) * 0.35
    clamped[:, 0] = np.clip(clamped[:, 0], -max_shift, max_shift)
    clamped[:, 1] = np.clip(clamped[:, 1], -max_shift, max_shift)
    clamped[:, 2] = np.clip(clamped[:, 2], -math.radians(25), math.radians(25))
    clamped[:, 3] = np.clip(clamped[:, 3], -0.15, 0.15)
    return clamped


def smoothing_params(method, strength, frame_count):
    strength = max(1, int(round(strength)))
    if method == "moving_average":
        return {"window": min(strength, frame_count)}
    if method == "savgol":
        return {"window": min(strength, frame_count), "polyorder": 3}
    if method == "gaussian":
        return {"sigma": max(0.1, strength / 10.0)}
    if method == "spline":
        return {"smoothing_factor": float(strength)}
    return {"window": min(strength, frame_count)}


def output_size(choice, width, height):
    if choice == "1080p":
        return 1920, 1080
    if choice == "720p":
        return 1280, 720
    if choice == "480p":
        return 854, 480
    return width, height


def build_affine_matrix(motion):
    dx, dy, angle, scale_delta = motion
    scale = float(np.clip(1.0 + scale_delta, 0.85, 1.15))
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale
    return np.array([[cos_a, -sin_a, dx], [sin_a, cos_a, dy]], dtype=np.float32)


def build_base_maps(width, height):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return xx, yy


def warp_with_mesh(frame, mesh_motion, base_x, base_y):
    height, width = frame.shape[:2]
    corr_x = cv2.resize(mesh_motion[:, :, 0].astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    corr_y = cv2.resize(mesh_motion[:, :, 1].astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    map_x = base_x - corr_x
    map_y = base_y - corr_y
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def resize_and_zoom(frame, out_width, out_height, crop_percent):
    if (frame.shape[1], frame.shape[0]) != (out_width, out_height):
        frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_LANCZOS4)

    crop_percent = float(np.clip(crop_percent, 0.0, 45.0))
    if crop_percent <= 0:
        return frame

    zoom = 1.0 / max(0.01, 1.0 - crop_percent / 100.0)
    zoom_width = max(out_width, int(round(out_width * zoom)))
    zoom_height = max(out_height, int(round(out_height * zoom)))
    zoomed = cv2.resize(frame, (zoom_width, zoom_height), interpolation=cv2.INTER_LANCZOS4)
    x = (zoom_width - out_width) // 2
    y = (zoom_height - out_height) // 2
    return zoomed[y:y + out_height, x:x + out_width]


def start_ffmpeg_writer(args, out_width, out_height, fps):
    fps = fps if fps and fps > 0 else 30.0
    cmd = [
        args.ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_width}x{out_height}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-i", args.input,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        args.output,
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    stderr_chunks = []

    def drain_stderr(pipe):
        try:
            for chunk in pipe:
                stderr_chunks.append(chunk)
        finally:
            pipe.close()

    thread = threading.Thread(target=drain_stderr, args=(proc.stderr,), daemon=True)
    thread.start()
    return proc, thread, stderr_chunks


def finalize_ffmpeg(proc, thread, stderr_chunks):
    if proc.stdin:
        proc.stdin.close()
    proc.wait()
    thread.join(timeout=10)
    if proc.returncode != 0:
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="ignore")
        last_lines = "\n".join(stderr_text.strip().splitlines()[-8:])
        emit_error(f"FFmpeg encoding failed (exit code {proc.returncode}):\n{last_lines}")
        return False
    return True


def process_video(args):
    model, device = load_raft_model(args.raft_model)
    if model is None:
        return None

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        emit_error(f"Cannot open video: {args.input}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 1 or width <= 0 or height <= 0:
        cap.release()
        emit_error("Could not determine usable video properties")
        return None

    max_edge = args.analysis_max_edge
    if max_edge <= 0:
        max_edge = 1280 if str(device) == "cuda" else 768

    mesh_grid = None if args.global_only else parse_mesh_grid(args.mesh_grid)
    if mesh_grid:
        emit("flow", 5, message=f"Estimating RAFT mesh flow at max edge {max_edge}px")
    else:
        emit("flow", 5, message=f"Estimating optical flow at max edge {max_edge}px")

    transforms = [np.zeros(4, dtype=np.float64)]
    mesh_motions = [np.zeros((mesh_grid[0], mesh_grid[1], 2), dtype=np.float64)] if mesh_grid else None
    previous_valid_motion = np.zeros(4, dtype=np.float64)
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            flow, analysis_prev, analysis_curr, scale_x, scale_y = estimate_flow(
                model,
                device,
                prev_frame,
                frame,
                args.max_iterations,
                max_edge,
            )
            motion, inliers, total, dense_matrix = dense_flow_to_motion(flow, scale_x, scale_y)
            mesh_motion = mesh_motion_from_flow(
                flow,
                dense_matrix,
                scale_x,
                scale_y,
                mesh_grid[0],
                mesh_grid[1],
            ) if mesh_grid else None

            if motion is None or not valid_motion(motion, width, height, inliers, total):
                fallback, fallback_inliers, fallback_total = sparse_fallback_motion(
                    analysis_prev,
                    analysis_curr,
                    scale_x,
                    scale_y,
                )
                if fallback is not None and valid_motion(fallback, width, height, fallback_inliers, fallback_total):
                    motion = fallback
                    inliers = fallback_inliers
                    total = fallback_total
                    previous_valid_motion = motion
                else:
                    motion = previous_valid_motion * 0.6
            else:
                previous_valid_motion = motion

            transforms.append(motion)
            if mesh_motions is not None:
                mesh_motions.append(mesh_motion)

            progress = 5 + (frame_idx / total_frames) * 85
            if frame_idx % 5 == 0:
                emit("flow", progress, message=f"Flow {frame_idx}/{total_frames} - inliers {inliers}/{total}")

        prev_frame = frame.copy()
        frame_idx += 1

    cap.release()

    if len(transforms) <= 1:
        emit_error("No frame motion was detected")
        return None

    transforms = clamp_transforms(np.array(transforms, dtype=np.float64), width, height)
    trajectory = np.cumsum(transforms, axis=0)

    emit("trajectory", 91, message=f"Smoothing trajectory with {args.smoothing_method}")
    try:
        smooth_trajectory = apply_smoothing(
            trajectory,
            args.smoothing_method,
            **smoothing_params(args.smoothing_method, args.smoothing_strength, len(transforms)),
        )
    except Exception as exc:
        emit_error(f"Smoothing failed: {exc}")
        return None

    correction = smooth_trajectory - trajectory
    stabilized_transforms = clamp_transforms(transforms + correction, width, height)
    mesh_stabilized = None
    if mesh_motions is not None and len(mesh_motions) == len(transforms):
        try:
            mesh_motions_array = np.array(mesh_motions, dtype=np.float64)
            mesh_trajectory = np.cumsum(mesh_motions_array, axis=0)
            mesh_flat = mesh_trajectory.reshape(mesh_trajectory.shape[0], -1)
            smooth_mesh_flat = apply_smoothing(
                mesh_flat,
                args.smoothing_method,
                **smoothing_params(args.smoothing_method, args.smoothing_strength, len(transforms)),
            )
            smooth_mesh_trajectory = smooth_mesh_flat.reshape(mesh_trajectory.shape)
            mesh_correction = smooth_mesh_trajectory - mesh_trajectory
            mesh_stabilized = mesh_motions_array + mesh_correction
        except Exception as exc:
            emit("trajectory", 92, message=f"Mesh smoothing unavailable, using global path: {exc}")
            mesh_stabilized = None

    max_corr = np.max(np.abs(correction), axis=0)
    emit(
        "trajectory",
        92,
        message=(
            "Trajectory smoothed - "
            f"max correction dx={max_corr[0]:.1f} dy={max_corr[1]:.1f} angle={max_corr[2]:.4f}"
        ),
    )

    out_width, out_height = output_size(args.resolution, width, height)
    emit("transform", 92, message=f"Encoding {out_width}x{out_height} at {fps:.1f} fps")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        emit_error("Failed to reopen video for stabilization")
        return None

    try:
        ffmpeg_proc, stderr_thread, stderr_chunks = start_ffmpeg_writer(args, out_width, out_height, fps)
    except Exception as exc:
        cap.release()
        emit_error(f"Failed to start FFmpeg: {exc}")
        return None

    frame_idx = 0
    frame_count = len(stabilized_transforms)
    base_x, base_y = build_base_maps(width, height) if mesh_stabilized is not None else (None, None)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        motion_idx = min(frame_idx, frame_count - 1)
        if mesh_stabilized is not None:
            stabilized = warp_with_mesh(frame, mesh_stabilized[motion_idx], base_x, base_y)
        else:
            motion = stabilized_transforms[motion_idx]
            matrix = build_affine_matrix(motion)
            stabilized = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
        output_frame = resize_and_zoom(stabilized, out_width, out_height, args.crop_percent)

        try:
            ffmpeg_proc.stdin.write(output_frame.tobytes())
        except BrokenPipeError:
            cap.release()
            emit_error("FFmpeg pipe closed unexpectedly")
            return None

        progress = 92 + (frame_idx / total_frames) * 8
        if frame_idx % 5 == 0:
            emit("transform", progress, message=f"Stabilizing {frame_idx}/{total_frames}")
        frame_idx += 1

    cap.release()

    emit("transform", 99, message="Finalizing FFmpeg encoding...")
    if not finalize_ffmpeg(ffmpeg_proc, stderr_thread, stderr_chunks):
        return None

    emit("transform", 100, message="Stabilization complete")
    return {
        "ok": True,
        "outputSize": os.path.getsize(args.output) if os.path.exists(args.output) else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="RAFT dense motion stabilization")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output video file")
    parser.add_argument("--ffmpeg", required=True, help="Path to FFmpeg")
    parser.add_argument(
        "--raft-model",
        default="raft-sintel",
        choices=["raft-sintel", "raft-things"],
        help="RAFT model variant",
    )
    parser.add_argument("--max-iterations", type=int, default=20, help="RAFT refinement iterations")
    parser.add_argument(
        "--smoothing-method",
        default="savgol",
        choices=["moving_average", "savgol", "gaussian", "spline"],
        help="Trajectory smoothing method",
    )
    parser.add_argument("--smoothing-strength", type=int, default=60, help="Smoothing strength")
    parser.add_argument("--crop-percent", type=int, default=10, help="Border crop percentage")
    parser.add_argument(
        "--resolution",
        default="source",
        choices=["source", "1080p", "720p", "480p"],
        help="Output resolution",
    )
    parser.add_argument(
        "--analysis-max-edge",
        type=int,
        default=0,
        help="Maximum long edge for RAFT motion analysis; 0 chooses automatically",
    )
    parser.add_argument(
        "--mesh-grid",
        default=DEFAULT_MESH_GRID,
        help="Rows x columns for RAFT mesh-flow stabilization, or off",
    )
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Disable mesh-flow warping and use one global camera path",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        emit_error(f"Input file not found: {args.input}")
        return 1
    if not os.path.exists(args.ffmpeg):
        emit_error(f"FFmpeg not found: {args.ffmpeg}")
        return 1

    result = process_video(args)
    if result is None:
        return 1

    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        emit_error(f"Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
