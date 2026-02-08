#!/usr/bin/env python3
"""
Panorama / mosaic accumulation.

Accumulates video frames onto an expanding canvas using feature matching
and homography estimation to create either:
  - A single panoramic image
  - A stabilized panorama video with an expanded field of view

Communicates progress to the Electron host via JSON lines on stdout.
"""

import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np


# ------------------------------------------------------------------
# Progress helpers
# ------------------------------------------------------------------

def emit(phase: str, progress: float, **extra):
    msg = {"phase": phase, "progress": round(progress, 2), **extra}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def emit_error(message: str):
    sys.stderr.write(json.dumps({"error": message}) + "\n")
    sys.stderr.flush()


# ------------------------------------------------------------------
# Feature detection
# ------------------------------------------------------------------

def create_detector(name: str):
    if name == "sift":
        return cv2.SIFT_create(nfeatures=2000)
    else:
        return cv2.ORB_create(nfeatures=3000)


def create_matcher(name: str):
    if name == "sift":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def match_features(det, matcher, gray1, gray2, ratio_thresh=0.75):
    """Detect and match features between two grayscale frames.
    Returns list of (pt1, pt2) matched point pairs."""
    kp1, des1 = det.detectAndCompute(gray1, None)
    kp2, des2 = det.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return []

    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_thresh * n.distance:
                good.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))
    return good


def estimate_homography(matches, ransac_thresh=4.0):
    """Estimate homography from matched point pairs using RANSAC."""
    if len(matches) < 4:
        return None
    pts1 = np.float32([m[0] for m in matches])
    pts2 = np.float32([m[1] for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None
    inliers = mask.ravel().sum() if mask is not None else 0
    if inliers < 4:
        return None
    return H


# ------------------------------------------------------------------
# Trajectory smoothing (for video output)
# ------------------------------------------------------------------

def decompose_homographies(Hs):
    """Extract translation and rotation from a sequence of 3x3 homographies.
    Returns Nx3 array of [dx, dy, da]."""
    n = len(Hs)
    transforms = np.zeros((n, 3), dtype=np.float64)
    for i, H in enumerate(Hs):
        dx = H[0, 2]
        dy = H[1, 2]
        da = math.atan2(H[1, 0], H[0, 0])
        transforms[i] = [dx, dy, da]
    return transforms


def smooth_transforms(transforms, radius):
    """Moving-average smooth of Nx3 transform array."""
    if radius < 1:
        return transforms.copy()
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    smoothed = np.copy(transforms)
    for col in range(3):
        padded = np.pad(transforms[:, col], radius, mode="edge")
        smoothed[:, col] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def recompose_homography(dx, dy, da):
    """Build a 3x3 affine-like homography from dx, dy, rotation angle."""
    cos_a = math.cos(da)
    sin_a = math.sin(da)
    return np.array([
        [cos_a, -sin_a, dx],
        [sin_a, cos_a, dy],
        [0, 0, 1],
    ], dtype=np.float64)


# ------------------------------------------------------------------
# Canvas computation
# ------------------------------------------------------------------

def compute_canvas_bounds(width, height, cumulative_Hs):
    """Given original frame size and cumulative homographies, compute
    the bounding box of the full mosaic canvas."""
    corners = np.float32([
        [0, 0], [width, 0], [width, height], [0, height]
    ]).reshape(-1, 1, 2)

    all_min = np.array([0.0, 0.0])
    all_max = np.array([float(width), float(height)])

    for H in cumulative_Hs:
        warped = cv2.perspectiveTransform(corners, H)
        pts = warped.reshape(-1, 2)
        all_min = np.minimum(all_min, pts.min(axis=0))
        all_max = np.maximum(all_max, pts.max(axis=0))

    return all_min, all_max


# ------------------------------------------------------------------
# Image mosaic output
# ------------------------------------------------------------------

def build_image_mosaic(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        emit_error(f"Cannot open video: {args.input}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < 2:
        emit_error("Video is too short to create a mosaic")
        sys.exit(1)

    det = create_detector(args.detector)
    matcher = create_matcher(args.detector)

    # --- Pass 1: compute pairwise homographies ---
    emit("features", 0)
    pair_Hs = []  # H[i] maps frame i to frame i+1
    prev_gray = None

    # Sample frames (every Nth for speed; use all if short)
    sample_step = max(1, total_frames // 300)

    sampled_frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % sample_step != 0 and i != total_frames - 1:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sampled_frames.append((i, frame, gray))
        if i % 20 == 0:
            emit("features", (i / total_frames) * 30)

    cap.release()
    emit("features", 30)

    # --- Match consecutive sampled frames ---
    emit("homography", 30)
    cumulative_Hs = [np.eye(3, dtype=np.float64)]  # identity for first frame

    for idx in range(1, len(sampled_frames)):
        _, _, g1 = sampled_frames[idx - 1]
        _, _, g2 = sampled_frames[idx]
        matches = match_features(det, matcher, g1, g2)
        H = estimate_homography(matches)
        if H is None:
            # No good match — assume identity (no motion)
            H = np.eye(3, dtype=np.float64)
        # H maps points from frame idx to frame idx-1
        # We want cumulative: map frame idx to frame 0
        cumulative = cumulative_Hs[-1] @ np.linalg.inv(H)
        cumulative_Hs.append(cumulative)
        if idx % 5 == 0:
            emit("homography", 30 + (idx / len(sampled_frames)) * 20)

    emit("homography", 50)

    # --- Compute canvas ---
    emit("compositing", 50)
    canvas_min, canvas_max = compute_canvas_bounds(width, height, cumulative_Hs)

    # Limit canvas to reasonable size (max 16k)
    canvas_w = int(canvas_max[0] - canvas_min[0])
    canvas_h = int(canvas_max[1] - canvas_min[1])
    max_dim = 16000
    if canvas_w > max_dim or canvas_h > max_dim:
        scale = min(max_dim / canvas_w, max_dim / canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)
    else:
        scale = 1.0

    # Translation to shift everything into positive coordinates
    offset = np.eye(3, dtype=np.float64)
    offset[0, 2] = -canvas_min[0] * scale
    offset[1, 2] = -canvas_min[1] * scale

    if scale != 1.0:
        scale_mat = np.diag([scale, scale, 1.0])
    else:
        scale_mat = np.eye(3)

    # --- Composite frames onto canvas ---
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for idx, (_, frame, _) in enumerate(sampled_frames):
        H_total = offset @ scale_mat @ cumulative_Hs[idx]
        warped = cv2.warpPerspective(frame, H_total, (canvas_w, canvas_h))

        if args.blend == "average":
            mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
            mask_f = mask.astype(np.float32)
            for c in range(3):
                canvas[:, :, c] = np.where(
                    mask,
                    ((canvas[:, :, c].astype(np.float32) * weight + warped[:, :, c].astype(np.float32))
                     / (weight + mask_f + 1e-8)).astype(np.uint8),
                    canvas[:, :, c],
                )
            weight += mask_f
        else:
            # Overlay (last-write-wins)
            mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
            canvas[mask] = warped[mask]

        if idx % 3 == 0:
            emit("compositing", 50 + (idx / len(sampled_frames)) * 45)

    emit("compositing", 95)

    # --- Save ---
    cv2.imwrite(args.output, canvas)
    emit("compositing", 100)

    out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0
    result = {"ok": True, "outputSize": out_size}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


# ------------------------------------------------------------------
# Video panorama output
# ------------------------------------------------------------------

def build_video_panorama(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        emit_error(f"Cannot open video: {args.input}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < 2:
        emit_error("Video is too short")
        sys.exit(1)

    det = create_detector(args.detector)
    matcher = create_matcher(args.detector)

    # --- Pass 1: compute pairwise homographies ---
    emit("features", 0)
    frames_data = []  # (gray,)
    pair_Hs = []

    ret, first_frame = cap.read()
    if not ret:
        emit_error("Cannot read first frame")
        sys.exit(1)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    for i in range(1, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        matches = match_features(det, matcher, prev_gray, gray)
        H = estimate_homography(matches)
        if H is None:
            H = np.eye(3, dtype=np.float64)
        pair_Hs.append(H)  # maps prev -> cur
        prev_gray = gray

        if i % 10 == 0:
            emit("features", (i / total_frames) * 30)

    cap.release()
    emit("features", 30)

    # --- Cumulative transforms ---
    emit("homography", 30)
    n = len(pair_Hs)
    raw_transforms = decompose_homographies(pair_Hs)

    # Cumulative sum
    cumulative = np.cumsum(raw_transforms, axis=0)
    # Prepend zero for first frame
    cumulative = np.vstack([np.zeros((1, 3)), cumulative])

    # Smooth
    smoothed = smooth_transforms(cumulative, args.smoothing)

    # Correction = smoothed - original
    corrections = smoothed - cumulative
    emit("homography", 50)

    # --- Pass 2: apply stabilizing warp ---
    emit("encoding", 50)
    cap = cv2.VideoCapture(args.input)

    # Pad canvas slightly to accommodate corrections
    pad = 50
    out_w = width + 2 * pad
    out_h = height + 2 * pad

    ffmpeg_cmd = [
        args.ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        args.output,
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for i in range(min(total_frames, len(corrections))):
        ret, frame = cap.read()
        if not ret:
            break

        dx, dy, da = corrections[i]
        H_corr = recompose_homography(dx, dy, da)
        # Shift to centre of padded canvas
        H_offset = np.eye(3, dtype=np.float64)
        H_offset[0, 2] = pad
        H_offset[1, 2] = pad
        H_total = H_offset @ H_corr

        warped = cv2.warpPerspective(frame, H_total, (out_w, out_h),
                                      borderMode=cv2.BORDER_REPLICATE)
        proc.stdin.write(warped.tobytes())

        if i % 10 == 0:
            emit("encoding", 50 + (i / total_frames) * 50)

    proc.stdin.close()
    proc.wait()
    cap.release()

    emit("encoding", 100)

    out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0
    result = {"ok": True, "outputSize": out_size}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Panorama / mosaic builder")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--type", choices=["video", "image"], default="video",
                        help="Output type")
    parser.add_argument("--detector", choices=["orb", "sift"], default="orb",
                        help="Feature detector")
    parser.add_argument("--smoothing", type=int, default=30,
                        help="Smoothing radius (video mode)")
    parser.add_argument("--blend", choices=["overlay", "average"], default="overlay",
                        help="Blend mode (image mode)")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to FFmpeg")
    args = parser.parse_args()

    if args.type == "image":
        build_image_mosaic(args)
    else:
        build_video_panorama(args)


if __name__ == "__main__":
    main()
