#!/usr/bin/env python3
"""
OpenCV feature-tracking video stabilization.

This path estimates global camera motion from sparse tracked points:
1. Detect stable points in the previous frame.
2. Track them into the current frame with pyramidal Lucas-Kanade optical flow.
3. Reject bad tracks with a forward/backward check.
4. Estimate a limited affine camera transform with RANSAC.
5. Smooth the camera trajectory, warp frames, and mux source audio.
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

from smoothing_lib import apply_smoothing


MIN_TRACKS = 12


def emit(phase, progress, **extra):
    """Emit progress as JSON to stdout."""
    msg = {"phase": phase, "progress": round(progress, 2), **extra}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def emit_error(message):
    """Emit error message to stderr."""
    sys.stderr.write(json.dumps({"error": message}) + "\n")
    sys.stderr.flush()


def create_keypoint_detector(detector_type, max_features):
    detector_type = detector_type.lower()
    max_features = max(100, int(max_features))

    if detector_type == "sift":
        return cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.01)
    if detector_type == "orb":
        return cv2.ORB_create(nfeatures=max_features, fastThreshold=12)
    if detector_type == "akaze":
        return cv2.AKAZE_create()
    if detector_type == "gftt":
        return None
    raise ValueError(f"Unknown detector type: {detector_type}")


def good_features(gray, max_features):
    """Detect a grid-distributed set of corners instead of one clustered set."""
    max_features = max(100, int(max_features))
    height, width = gray.shape[:2]
    grid_rows = 4
    grid_cols = 6
    per_cell = max(12, int(math.ceil(max_features / float(grid_rows * grid_cols))))
    points = []

    for row in range(grid_rows):
        y0 = int(round(row * height / grid_rows))
        y1 = int(round((row + 1) * height / grid_rows))
        for col in range(grid_cols):
            x0 = int(round(col * width / grid_cols))
            x1 = int(round((col + 1) * width / grid_cols))
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            cell_pts = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=per_cell,
                qualityLevel=0.008,
                minDistance=7,
                blockSize=7,
                useHarrisDetector=False,
            )
            if cell_pts is None:
                continue
            cell_pts = cell_pts.reshape(-1, 2)
            cell_pts[:, 0] += x0
            cell_pts[:, 1] += y0
            points.append(cell_pts)

    if points:
        pts = np.vstack(points).astype(np.float32)
        if len(pts) > max_features:
            pts = pts[:max_features]
        return pts.reshape(-1, 1, 2)

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        useHarrisDetector=False,
    )
    if pts is None:
        return None
    return pts.astype(np.float32)


def detect_tracking_points(gray, detector_type, max_features):
    """Return Nx1x2 points suitable for calcOpticalFlowPyrLK."""
    detector_type = detector_type.lower()
    max_features = max(100, int(max_features))

    if detector_type == "gftt":
        return good_features(gray, max_features)

    detector = create_keypoint_detector(detector_type, max_features)
    keypoints = detector.detect(gray, None)
    if keypoints:
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_features]
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    else:
        pts = None

    if pts is None or len(pts) < MIN_TRACKS:
        return good_features(gray, max_features)
    return pts


def track_points(prev_gray, curr_gray, prev_pts):
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=1e-4,
    )

    curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    if curr_pts is None or status is None:
        return None, None

    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, curr_pts, None, **lk_params)
    if back_pts is None or back_status is None:
        return None, None

    fb_error = np.linalg.norm(prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
    valid = (status.ravel() == 1) & (back_status.ravel() == 1) & (fb_error < 1.5)
    if np.count_nonzero(valid) < MIN_TRACKS:
        return None, None

    return prev_pts.reshape(-1, 2)[valid], curr_pts.reshape(-1, 2)[valid]


def estimate_camera_transform(prev_pts, curr_pts, transform_type):
    """Estimate a stable 4-DOF affine camera motion from tracked points."""
    if prev_pts is None or curr_pts is None or len(prev_pts) < MIN_TRACKS:
        return None, 0, 0

    source_pts = prev_pts
    target_pts = curr_pts

    if transform_type == "homography" and len(prev_pts) >= 16:
        homography, homography_inliers = cv2.findHomography(
            prev_pts,
            curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=3000,
            confidence=0.995,
        )
        if homography is not None and homography_inliers is not None:
            mask = homography_inliers.ravel().astype(bool)
            if np.count_nonzero(mask) >= MIN_TRACKS:
                source_pts = prev_pts[mask]
                target_pts = curr_pts[mask]

    matrix, inliers = cv2.estimateAffinePartial2D(
        source_pts,
        target_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        return None, 0, len(prev_pts)

    return matrix, int(np.count_nonzero(inliers)), len(source_pts)


def decompose_affine(matrix):
    """Return [dx, dy, angle, scale_delta] from a 2x3 affine matrix."""
    if matrix is None:
        return np.zeros(4, dtype=np.float64)

    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    angle = float(math.atan2(matrix[1, 0], matrix[0, 0]))
    scale = float(math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    return np.array([dx, dy, angle, scale - 1.0], dtype=np.float64)


def refine_with_ecc(prev_gray, curr_gray, matrix):
    """Use ECC image alignment as an expensive refinement of the RANSAC estimate."""
    if matrix is None:
        return None

    height, width = prev_gray.shape[:2]
    scale = min(1.0, 720.0 / float(max(height, width)))
    if scale < 1.0:
        prev_small = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        prev_small = prev_gray
        curr_small = curr_gray

    warp = matrix.astype(np.float32).copy()
    warp[0, 2] *= scale
    warp[1, 2] *= scale

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        _cc, refined = cv2.findTransformECC(
            prev_small,
            curr_small,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error:
        return matrix

    refined = refined.astype(np.float64)
    refined[0, 2] /= scale
    refined[1, 2] /= scale
    if not np.isfinite(refined).all():
        return matrix
    return refined


def valid_motion(motion, width, height, inliers, total):
    if total < MIN_TRACKS or inliers < MIN_TRACKS:
        return False

    inlier_ratio = inliers / max(total, 1)
    if inlier_ratio < 0.35:
        return False

    max_shift = max(width, height) * 0.35
    if abs(motion[0]) > max_shift or abs(motion[1]) > max_shift:
        return False
    if abs(motion[2]) > math.radians(25):
        return False
    if abs(motion[3]) > 0.20:
        return False
    return True


def clamp_transforms(transforms, width, height):
    clamped = np.array(transforms, dtype=np.float64, copy=True)
    max_shift = max(width, height) * 0.35
    clamped[:, 0] = np.clip(clamped[:, 0], -max_shift, max_shift)
    clamped[:, 1] = np.clip(clamped[:, 1], -max_shift, max_shift)
    clamped[:, 2] = np.clip(clamped[:, 2], -math.radians(25), math.radians(25))
    clamped[:, 3] = np.clip(clamped[:, 3], -0.15, 0.15)
    return clamped


def dense_flow_to_motion(flow, scale_x, scale_y):
    height, width = flow.shape[:2]
    step = max(4, min(height, width) // 70)
    ys, xs = np.mgrid[0:height:step, 0:width:step]
    pts1 = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    flow_sub = flow[0:height:step, 0:width:step].reshape(-1, 2).astype(np.float32)
    finite = np.isfinite(flow_sub).all(axis=1)
    pts1 = pts1[finite]
    flow_sub = flow_sub[finite]
    if len(pts1) < MIN_TRACKS:
        return None, 0, 0

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
        return None, 0, len(pts1)

    motion = decompose_affine(matrix)
    motion[0] *= scale_x
    motion[1] *= scale_y
    return motion, int(np.count_nonzero(inliers)), len(pts1)


def farneback_fallback_motion(prev_gray, curr_gray):
    height, width = prev_gray.shape[:2]
    scale = min(1.0, 720.0 / float(max(height, width)))
    if scale < 1.0:
        prev_small = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        prev_small = prev_gray
        curr_small = curr_gray

    flow = cv2.calcOpticalFlowFarneback(
        prev_small,
        curr_small,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=25,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    return dense_flow_to_motion(flow, 1.0 / scale, 1.0 / scale)


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
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        emit_error(f"Failed to open input video: {args.input}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 1 or width <= 0 or height <= 0:
        emit_error("Could not determine usable video properties")
        cap.release()
        return 1

    emit("features", 0, message="Initializing feature tracker")
    try:
        create_keypoint_detector(args.detector, args.max_features)
    except ValueError as exc:
        emit_error(str(exc))
        cap.release()
        return 1

    transforms = [np.zeros(4, dtype=np.float64)]
    prev_gray = None
    previous_valid_motion = np.zeros(4, dtype=np.float64)
    frame_idx = 0

    emit("features", 0, message=f"Tracking camera motion with {args.detector.upper()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.equalizeHist(curr_gray)

        if prev_gray is not None:
            prev_pts = detect_tracking_points(prev_gray, args.detector, args.max_features)
            motion = None
            inliers = 0
            total = 0

            if prev_pts is not None and len(prev_pts) >= MIN_TRACKS:
                tracked_prev, tracked_curr = track_points(prev_gray, curr_gray, prev_pts)
                matrix, inliers, total = estimate_camera_transform(
                    tracked_prev,
                    tracked_curr,
                    args.transform_type,
                )
                if matrix is not None:
                    if args.ecc_refine:
                        matrix = refine_with_ecc(prev_gray, curr_gray, matrix)
                    candidate = decompose_affine(matrix)
                    if valid_motion(candidate, width, height, inliers, total):
                        motion = candidate

            if motion is None and args.dense_fallback:
                dense_motion, dense_inliers, dense_total = farneback_fallback_motion(prev_gray, curr_gray)
                if dense_motion is not None and valid_motion(dense_motion, width, height, dense_inliers, dense_total):
                    motion = dense_motion
                    inliers = dense_inliers
                    total = dense_total

            if motion is None:
                motion = previous_valid_motion * 0.6
            else:
                previous_valid_motion = motion

            transforms.append(motion)

        prev_gray = curr_gray
        frame_idx += 1

        progress = (frame_idx / total_frames) * 80
        if frame_idx % 5 == 0 or frame_idx == total_frames:
            emit("features", progress, message=f"Frame {frame_idx}/{total_frames} - inliers {inliers}/{total}")

    cap.release()

    frame_count = len(transforms)
    if frame_count <= 1:
        emit_error("No frame motion was detected")
        return 1

    transforms = clamp_transforms(np.array(transforms, dtype=np.float64), width, height)
    trajectory = np.cumsum(transforms, axis=0)

    emit("trajectory", 80, message=f"Smoothing trajectory with {args.smoothing_method}")
    try:
        smooth_trajectory = apply_smoothing(
            trajectory,
            args.smoothing_method,
            **smoothing_params(args.smoothing_method, args.smoothing_strength, frame_count),
        )
    except Exception as exc:
        emit_error(f"Smoothing failed: {exc}")
        return 1

    correction = smooth_trajectory - trajectory
    stabilized_transforms = clamp_transforms(transforms + correction, width, height)
    max_correction = np.max(np.abs(correction), axis=0)
    emit(
        "trajectory",
        85,
        message=(
            "Trajectory smoothed - "
            f"max correction dx={max_correction[0]:.1f} dy={max_correction[1]:.1f} "
            f"angle={max_correction[2]:.4f}"
        ),
    )

    out_width, out_height = output_size(args.resolution, width, height)
    emit("transform", 85, message=f"Encoding {out_width}x{out_height} at {fps:.1f} fps")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        emit_error("Failed to reopen video for stabilization")
        return 1

    try:
        ffmpeg_proc, stderr_thread, stderr_chunks = start_ffmpeg_writer(args, out_width, out_height, fps)
    except Exception as exc:
        cap.release()
        emit_error(f"Failed to start FFmpeg: {exc}")
        return 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        motion = stabilized_transforms[min(frame_idx, frame_count - 1)]
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
            return 1

        frame_idx += 1
        progress = 85 + (frame_idx / total_frames) * 15
        if frame_idx % 5 == 0 or frame_idx == total_frames:
            emit("transform", progress, message=f"Encoding frame {frame_idx}/{total_frames}")

    cap.release()

    emit("transform", 99, message="Finalizing FFmpeg encoding...")
    if not finalize_ffmpeg(ffmpeg_proc, stderr_thread, stderr_chunks):
        return 1

    emit("transform", 100, message="Stabilization complete")
    result = {
        "ok": True,
        "outputSize": os.path.getsize(args.output) if os.path.exists(args.output) else 0,
    }
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


def main():
    parser = argparse.ArgumentParser(description="OpenCV feature tracking stabilization")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output video file")
    parser.add_argument("--ffmpeg", required=True, help="Path to FFmpeg executable")
    parser.add_argument(
        "--detector",
        default="gftt",
        choices=["gftt", "sift", "orb", "akaze"],
        help="Feature detector type",
    )
    parser.add_argument("--max-features", type=int, default=3000, help="Maximum features to detect")
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=["affine", "homography"],
        help="Robust transform estimation type",
    )
    parser.add_argument(
        "--smoothing-method",
        default="savgol",
        choices=["moving_average", "savgol", "gaussian", "spline"],
        help="Trajectory smoothing method",
    )
    parser.add_argument("--smoothing-strength", type=float, default=70, help="Smoothing strength")
    parser.add_argument("--crop-percent", type=float, default=10, help="Border crop percentage")
    parser.add_argument(
        "--resolution",
        default="source",
        choices=["source", "1080p", "720p", "480p"],
        help="Output resolution",
    )
    parser.add_argument(
        "--no-ecc-refine",
        dest="ecc_refine",
        action="store_false",
        help="Disable ECC refinement after sparse tracking",
    )
    parser.add_argument(
        "--no-dense-fallback",
        dest="dense_fallback",
        action="store_false",
        help="Disable Farneback dense-flow fallback",
    )
    parser.set_defaults(ecc_refine=True, dense_fallback=True)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        emit_error(f"Input file not found: {args.input}")
        return 1
    if not os.path.exists(args.ffmpeg):
        emit_error(f"FFmpeg not found: {args.ffmpeg}")
        return 1

    try:
        return process_video(args)
    except Exception as exc:
        emit_error(f"Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
