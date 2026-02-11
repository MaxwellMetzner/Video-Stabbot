#!/usr/bin/env python3
"""
OpenCV Feature Tracking Video Stabilization

Uses feature detection (SIFT/ORB/AKAZE) + homography estimation + trajectory smoothing
to stabilize video without relying on gyroscope data.
"""

import argparse
import json
import sys
import os
import subprocess
import threading
import cv2
import numpy as np
from pathlib import Path

# Import smoothing library
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


def create_feature_detector(detector_type):
    """Create feature detector based on type"""
    detector_type = detector_type.lower()

    if detector_type == 'sift':
        return cv2.SIFT_create()
    elif detector_type == 'orb':
        return cv2.ORB_create(nfeatures=3000)
    elif detector_type == 'akaze':
        return cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def detect_and_match_features(prev_gray, curr_gray, detector, max_features=2000):
    """
    Detect features in both frames and match them

    Returns:
        matched_prev_pts: Nx2 array of matched points in previous frame
        matched_curr_pts: Nx2 array of matched points in current frame
    """
    # Detect keypoints and compute descriptors
    kp1, des1 = detector.detectAndCompute(prev_gray, None)
    kp2, des2 = detector.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None

    # Match features using BFMatcher or FLANN
    if isinstance(detector, cv2.ORB):
        # Use Hamming distance for ORB (binary descriptors)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        # Use L2 distance for SIFT/AKAZE
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Find matches with k=2 for ratio test
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        return None, None

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 10:
        return None, None

    # Limit to max_features best matches
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_features]

    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2


def estimate_transform(pts1, pts2, transform_type='affine'):
    """
    Estimate transformation between matched points

    Args:
        pts1: Nx2 array of points in frame 1
        pts2: Nx2 array of points in frame 2
        transform_type: 'affine' or 'homography'

    Returns:
        transform: 2x3 affine or 3x3 homography matrix
        inliers: Boolean mask of inlier points
    """
    if len(pts1) < 4:
        return None, None

    if transform_type == 'affine':
        # Estimate affine transform (rotation + translation + scale)
        transform, inliers = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99
        )
    else:  # homography
        # Estimate full homography (includes perspective)
        transform, inliers = cv2.findHomography(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99
        )

    return transform, inliers


def decompose_affine_transform(transform):
    """
    Decompose 2x3 affine transform into [dx, dy, da]

    Args:
        transform: 2x3 affine matrix

    Returns:
        [dx, dy, da]: Translation x, translation y, rotation angle
    """
    if transform is None:
        return np.array([0.0, 0.0, 0.0])

    dx = transform[0, 2]
    dy = transform[1, 2]

    # Extract rotation angle from rotation matrix
    da = np.arctan2(transform[1, 0], transform[0, 0])

    return np.array([dx, dy, da])


def process_video(args):
    """Main video processing pipeline"""

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        emit_error(f"Failed to open input video: {args.input}")
        return 1

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        emit_error("Could not determine video frame count")
        return 1

    emit('features', 0, message="Initializing feature detector")

    # Create feature detector
    try:
        detector = create_feature_detector(args.detector)
    except ValueError as e:
        emit_error(str(e))
        return 1

    # ============================================================
    # Pass 1: Extract transforms between consecutive frames
    # ============================================================

    transforms = []
    prev_gray = None
    frame_idx = 0

    emit('features', 0, message=f"Detecting features with {args.detector.upper()} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Detect and match features
            pts1, pts2 = detect_and_match_features(
                prev_gray, curr_gray, detector, args.max_features
            )

            if pts1 is not None and pts2 is not None:
                # Estimate transform
                transform, inliers = estimate_transform(pts1, pts2, args.transform_type)

                if transform is not None:
                    # Decompose to [dx, dy, da]
                    if args.transform_type == 'affine':
                        motion = decompose_affine_transform(transform)
                    else:
                        # For homography, extract affine approximation
                        # Use top-left 2x3 submatrix
                        affine_approx = transform[:2, :]
                        motion = decompose_affine_transform(affine_approx)

                    transforms.append(motion)
                else:
                    # No transform found - assume no motion
                    transforms.append(np.array([0.0, 0.0, 0.0]))
            else:
                # No features matched - assume no motion
                transforms.append(np.array([0.0, 0.0, 0.0]))
        else:
            # First frame - no previous frame to compare
            transforms.append(np.array([0.0, 0.0, 0.0]))

        prev_gray = curr_gray
        frame_idx += 1

        # Update progress (0-80%)
        progress = (frame_idx / total_frames) * 80
        if frame_idx % 5 == 0 or frame_idx == total_frames:
            emit('features', progress, message=f"Frame {frame_idx}/{total_frames} — {len(transforms)} transforms")

    cap.release()
    emit('features', 80, message=f"Feature detection complete — {len(transforms)} transforms from {total_frames} frames")

    # Convert to numpy array
    transforms = np.array(transforms)

    # Build cumulative trajectory
    trajectory = np.cumsum(transforms, axis=0)

    emit('trajectory', 80, message=f"Computing smooth trajectory (method: {args.smoothing_method})")

    # ============================================================
    # Pass 2: Smooth the trajectory
    # ============================================================

    # Convert smoothing strength to appropriate parameters
    if args.smoothing_method == 'moving_average':
        smooth_params = {'window': int(args.smoothing_strength)}
    elif args.smoothing_method == 'savgol':
        window = int(args.smoothing_strength)
        # Ensure odd window
        if window % 2 == 0:
            window += 1
        smooth_params = {'window': window, 'polyorder': 3}
    elif args.smoothing_method == 'gaussian':
        smooth_params = {'sigma': args.smoothing_strength / 10.0}
    elif args.smoothing_method == 'spline':
        smooth_params = {'smoothing_factor': args.smoothing_strength}
    else:
        smooth_params = {'window': int(args.smoothing_strength)}

    emit('trajectory', 82, message=f"Applying {args.smoothing_method} smoothing (strength={args.smoothing_strength})")

    try:
        smooth_trajectory = apply_smoothing(trajectory, args.smoothing_method, **smooth_params)
    except Exception as e:
        emit_error(f"Smoothing failed: {str(e)}")
        return 1

    # Calculate correction transforms
    correction = smooth_trajectory - trajectory

    # Log trajectory statistics
    max_correction = np.max(np.abs(correction), axis=0)
    emit('trajectory', 85, message=f"Trajectory smoothed — max correction: dx={max_correction[0]:.1f} dy={max_correction[1]:.1f} da={max_correction[2]:.4f}")

    # ============================================================
    # Pass 3: Apply stabilization and encode
    # ============================================================

    # Reopen video for second pass
    cap = cv2.VideoCapture(args.input)

    # Determine output resolution
    if args.resolution == '1080p':
        out_width, out_height = 1920, 1080
    elif args.resolution == '720p':
        out_width, out_height = 1280, 720
    elif args.resolution == '480p':
        out_width, out_height = 854, 480
    else:  # source
        out_width, out_height = width, height

    # Calculate crop to remove black borders
    crop_percent = args.crop_percent / 100.0
    crop_w = int(out_width * (1 - crop_percent))
    crop_h = int(out_height * (1 - crop_percent))
    crop_x = (out_width - crop_w) // 2
    crop_y = (out_height - crop_h) // 2

    emit('transform', 85, message=f"Encoding with FFmpeg — output: {crop_w}x{crop_h} @ {fps:.1f}fps")

    # Use FFmpeg to encode (pipe frames in)
    ffmpeg_cmd = [
        args.ffmpeg,
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{crop_w}x{crop_h}',
        '-r', str(fps),
        '-i', '-',  # Read from stdin
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-an',  # No audio (raw pipe input has no audio stream)
        args.output
    ]

    try:
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        emit_error(f"Failed to start FFmpeg: {str(e)}")
        return 1

    # Drain FFmpeg stderr in a background thread to prevent pipe buffer deadlock
    ffmpeg_stderr_lines = []
    def _drain_stderr(pipe):
        try:
            for line in pipe:
                ffmpeg_stderr_lines.append(line)
        except Exception:
            pass
        finally:
            pipe.close()

    stderr_thread = threading.Thread(target=_drain_stderr, args=(ffmpeg_proc.stderr,), daemon=True)
    stderr_thread.start()

    frame_idx = 0
    write_errors = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get correction transform for this frame
        dx, dy, da = correction[frame_idx]

        # Build affine transform matrix
        transform_matrix = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy]
        ], dtype=np.float32)

        # Apply transform to stabilize
        stabilized = cv2.warpAffine(
            frame,
            transform_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Resize if needed
        if (out_width, out_height) != (width, height):
            stabilized = cv2.resize(stabilized, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

        # Crop to remove black borders
        cropped = stabilized[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        # Write frame to FFmpeg
        try:
            ffmpeg_proc.stdin.write(cropped.tobytes())
        except BrokenPipeError:
            emit_error("FFmpeg pipe closed unexpectedly")
            break
        except Exception as e:
            write_errors += 1
            if write_errors > 5:
                emit_error(f"Too many write errors: {str(e)}")
                break

        frame_idx += 1

        # Update progress (85-100%)
        progress = 85 + (frame_idx / total_frames) * 15
        if frame_idx % 5 == 0 or frame_idx == total_frames:
            emit('transform', progress, message=f"Encoding frame {frame_idx}/{total_frames}")

    cap.release()

    # Close FFmpeg stdin and wait for completion
    emit('transform', 99, message="Finalizing FFmpeg encoding…")
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    stderr_thread.join(timeout=10)

    if ffmpeg_proc.returncode != 0:
        stderr_text = b''.join(ffmpeg_stderr_lines).decode('utf-8', errors='ignore')
        last_lines = stderr_text.strip().split('\n')[-5:]
        emit_error(f"FFmpeg encoding failed (exit code {ffmpeg_proc.returncode}):\n{''.join(last_lines)}")
        return 1

    emit('transform', 100, message="Stabilization complete")

    # Get output file size
    output_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0

    # Emit final result
    result = {
        "ok": True,
        "outputSize": output_size
    }
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()

    return 0


def main():
    parser = argparse.ArgumentParser(description="OpenCV feature tracking stabilization")

    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output video file')
    parser.add_argument('--ffmpeg', required=True, help='Path to FFmpeg executable')
    parser.add_argument('--detector', default='sift', choices=['sift', 'orb', 'akaze'],
                       help='Feature detector type')
    parser.add_argument('--max-features', type=int, default=2000,
                       help='Maximum features to detect (500-5000)')
    parser.add_argument('--transform-type', default='affine', choices=['affine', 'homography'],
                       help='Transform estimation type')
    parser.add_argument('--smoothing-method', default='savgol',
                       choices=['moving_average', 'savgol', 'gaussian', 'spline'],
                       help='Trajectory smoothing method')
    parser.add_argument('--smoothing-strength', type=float, default=50,
                       help='Smoothing strength parameter (10-200)')
    parser.add_argument('--crop-percent', type=float, default=10,
                       help='Crop percentage to remove borders (0-20)')
    parser.add_argument('--resolution', default='source',
                       choices=['source', '1080p', '720p', '480p'],
                       help='Output resolution')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input):
        emit_error(f"Input file not found: {args.input}")
        return 1

    if not os.path.exists(args.ffmpeg):
        emit_error(f"FFmpeg not found: {args.ffmpeg}")
        return 1

    try:
        sys.exit(process_video(args))
    except Exception as e:
        emit_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    main()
