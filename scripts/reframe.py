#!/usr/bin/env python3
"""
Subject-locked reframing stabilization.

Tracks a subject (face or point) across video frames and produces a
smoothly reframed output that keeps the subject centred.  Communicates
progress to the Electron host via JSON lines on stdout.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


# ------------------------------------------------------------------
# Progress helpers
# ------------------------------------------------------------------

def emit(phase: str, progress: float, **extra):
    """Print a JSON progress line for the Electron host."""
    msg = {"phase": phase, "progress": round(progress, 2), **extra}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def emit_error(message: str):
    sys.stderr.write(json.dumps({"error": message}) + "\n")
    sys.stderr.flush()


# ------------------------------------------------------------------
# Face detection
# ------------------------------------------------------------------

_face_cascade = None


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(xml)
    return _face_cascade


def detect_face(gray_frame):
    """Return (cx, cy, w, h) of the largest face or None."""
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(gray_frame, 1.3, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    # Largest face by area
    areas = [w * h for (_, _, w, h) in faces]
    idx = int(np.argmax(areas))
    x, y, w, h = faces[idx]
    return (x + w // 2, y + h // 2, w, h)


# ------------------------------------------------------------------
# Point tracking (Lucas-Kanade)
# ------------------------------------------------------------------

_lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def track_point(prev_gray, cur_gray, prev_pt):
    """Track a single point forward.  Returns (x, y) or None."""
    pts = np.array([[prev_pt]], dtype=np.float32)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, pts, None, **_lk_params
    )
    if status[0][0] == 1:
        return tuple(new_pts[0][0].astype(int))
    return None


# ------------------------------------------------------------------
# Trajectory smoothing
# ------------------------------------------------------------------

def smooth_trajectory(traj, radius):
    """Moving-average smooth of an Nx2 trajectory."""
    if radius < 1:
        return traj.copy()
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    smoothed = np.copy(traj).astype(np.float64)
    for col in range(traj.shape[1]):
        padded = np.pad(traj[:, col].astype(np.float64), radius, mode="edge")
        smoothed[:, col] = np.convolve(padded, kernel, mode="valid")
    return smoothed


# ------------------------------------------------------------------
# Main processing
# ------------------------------------------------------------------

def process(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        emit_error(f"Cannot open video: {args.input}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < 2:
        emit_error("Video is too short to process")
        sys.exit(1)

    scale = max(0.2, min(1.0, args.scale / 100.0))
    crop_w = int(width * scale)
    crop_h = int(height * scale)
    # libx264 + yuv420p requires even dimensions
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    # Parse output resolution
    out_w, out_h = crop_w, crop_h
    if args.resolution != "same":
        try:
            parts = args.resolution.split("x")
            out_w, out_h = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            out_w, out_h = crop_w, crop_h
    # Ensure even dimensions for encoder
    out_w = out_w - (out_w % 2)
    out_h = out_h - (out_h % 2)

    # ---- Pass 1: tracking ----
    emit("tracking", 0)
    centres = []
    prev_gray = None
    prev_pt = None

    # How often (in frames) to re-run the expensive Haar cascade.
    # In between, fast LK optical-flow tracking is used instead.
    face_detect_interval = max(1, int(fps // 2))  # ~2 detections per second

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cx, cy = width // 2, height // 2  # fallback centre

        if args.mode == "face":
            # Use LK tracking most frames; only run Haar cascade on the
            # first frame, every face_detect_interval frames, or when
            # LK tracking has been lost.
            need_detect = (
                prev_pt is None
                or (i % face_detect_interval == 0)
            )

            tracked = None
            if prev_pt is not None and prev_gray is not None:
                tracked = track_point(prev_gray, gray, prev_pt)

            if need_detect or tracked is None:
                det = detect_face(gray)
                if det is not None:
                    cx, cy = det[0], det[1]
                    prev_pt = (cx, cy)
                elif tracked is not None:
                    cx, cy = tracked
                    prev_pt = (cx, cy)
                elif prev_pt is not None:
                    cx, cy = prev_pt
            else:
                cx, cy = tracked
                prev_pt = (cx, cy)
        else:
            # Point mode — use centre of frame for first frame, then track
            if prev_pt is None:
                prev_pt = (width // 2, height // 2)
                cx, cy = prev_pt
            else:
                tracked = track_point(prev_gray, gray, prev_pt) if prev_gray is not None else None
                if tracked:
                    cx, cy = tracked
                    prev_pt = (cx, cy)
                else:
                    cx, cy = prev_pt

        centres.append((cx, cy))
        prev_gray = gray.copy()

        if i % 10 == 0:
            emit("tracking", (i / total_frames) * 50)

    cap.release()
    emit("tracking", 50)

    # ---- Smooth trajectory ----
    traj = np.array(centres, dtype=np.float64)
    smoothed = smooth_trajectory(traj, args.smoothing)

    # ---- Pass 2: reframe and write ----
    emit("reframing", 50)

    # Use a temporary file for raw frames piped to FFmpeg
    cap = cv2.VideoCapture(args.input)

    # Build FFmpeg command to encode output
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
        stderr=subprocess.PIPE,
    )

    write_error = None
    for i in range(min(total_frames, len(smoothed))):
        ret, frame = cap.read()
        if not ret:
            break

        sx, sy = smoothed[i]
        sx = int(np.clip(sx, crop_w // 2, width - crop_w // 2))
        sy = int(np.clip(sy, crop_h // 2, height - crop_h // 2))

        x1 = sx - crop_w // 2
        y1 = sy - crop_h // 2
        crop = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]

        if crop.shape[1] != out_w or crop.shape[0] != out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        try:
            proc.stdin.write(crop.tobytes())
        except BrokenPipeError:
            write_error = "FFmpeg pipe closed unexpectedly"
            break

        if i % 10 == 0:
            emit("reframing", 50 + (i / total_frames) * 50)

    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    cap.release()

    if proc.returncode != 0 or write_error:
        ff_stderr = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
        detail = ff_stderr.split("\n")[-1] if ff_stderr else (write_error or "unknown error")
        emit_error(f"FFmpeg failed: {detail}")
        sys.exit(1)

    emit("reframing", 100)

    # Report result
    out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0
    result = {"ok": True, "outputSize": out_size}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Subject-locked reframing")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--mode", choices=["face", "point"], default="face")
    parser.add_argument("--smoothing", type=int, default=30, help="Smoothing radius")
    parser.add_argument("--scale", type=float, default=70, help="Crop scale percentage")
    parser.add_argument("--resolution", default="same", help="Output resolution WxH or 'same'")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to FFmpeg")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
