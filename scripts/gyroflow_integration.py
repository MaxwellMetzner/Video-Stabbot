#!/usr/bin/env python3
"""
Gyroflow CLI Integration for Video Stabbot

Wrapper script that detects gyroscope metadata in videos and executes
Gyroflow CLI for superior gyro-based stabilization.

Supports cameras: GoPro, DJI, Insta360, Sony, and others with embedded gyro data.
"""

import argparse
import json
import sys
import os
import subprocess
import tempfile
from pathlib import Path


def emit(phase, progress, **extra):
    """Emit progress as JSON to stdout"""
    msg = {"phase": phase, "progress": round(progress, 2), **extra}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def emit_error(message):
    """Emit error message to stderr"""
    sys.stderr.write(json.dumps({"error": message}) + "\n")
    sys.stderr.flush()


def check_gyro_metadata(ffprobe_path, video_path):
    """
    Check if video contains gyroscope metadata

    Returns:
        bool: True if gyro data detected, False otherwise
    """
    try:
        result = subprocess.run(
            [ffprobe_path, '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', video_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return False

        data = json.loads(result.stdout)

        # Check for common gyro metadata indicators
        # GoPro: GPMF stream
        # DJI: Custom metadata streams
        # Insta360: Gyro data streams

        for stream in data.get('streams', []):
            codec_name = stream.get('codec_name', '').lower()
            codec_tag = stream.get('codec_tag_string', '').lower()

            # GoPro GPMF
            if codec_name == 'bin_data' or codec_tag == 'gpmd':
                return True

            # DJI metadata
            if 'dji' in codec_tag or 'djmd' in codec_tag:
                return True

        # Check format tags for gyro indicators
        format_tags = data.get('format', {}).get('tags', {})
        for key, value in format_tags.items():
            key_lower = key.lower()
            if 'gyro' in key_lower or 'accelerometer' in key_lower or 'gpmf' in key_lower:
                return True

        return False

    except Exception as e:
        emit_error(f"Failed to check gyro metadata: {str(e)}")
        return False


def create_gyroflow_preset(settings):
    """
    Create a minimal Gyroflow preset JSON file

    Args:
        settings: Dictionary with stabilization parameters

    Returns:
        str: Path to created preset file, or None
    """
    preset = {
        "version": 1,
        "stabilization": {
            "method": "default",
            "smoothing_params": {
                "time_constant": settings.get('smoothing', 0.5)
            },
            "fov": 1.0 + (settings.get('fov', 0) / 100.0),
            "horizon_lock": {
                "enabled": settings.get('horizon_lock', False),
                "roll": 0.0
            }
        },
        "output": {
            "codec": "h264",
            "bitrate": 50.0,  # Mbps
            "resolution": settings.get('resolution', 'source')
        }
    }

    try:
        fd, preset_path = tempfile.mkstemp(suffix='.gyroflow', text=True)
        with os.fdopen(fd, 'w') as f:
            json.dump(preset, f, indent=2)
        return preset_path
    except Exception as e:
        emit_error(f"Failed to create preset: {str(e)}")
        return None


def run_gyroflow(gyroflow_path, input_path, output_path, settings, ffprobe_path):
    """
    Execute Gyroflow CLI for stabilization

    Args:
        gyroflow_path: Path to gyroflow-cli executable
        input_path: Input video path
        output_path: Output video path
        settings: Settings dictionary
        ffprobe_path: Path to ffprobe (for metadata checking)

    Returns:
        dict: Result dictionary with outputSize
    """
    emit('detect', 5, message="Checking for gyroscope data")

    # Check if video has gyro metadata
    has_gyro = check_gyro_metadata(ffprobe_path, input_path)

    if not has_gyro:
        emit_error(
            "No gyroscope data detected in this video.\n\n"
            "Gyroflow requires embedded gyro metadata from cameras like:\n"
            "• GoPro (Hero 5 and newer)\n"
            "• DJI (most drones and action cameras)\n"
            "• Insta360 (various models)\n"
            "• Sony (select cameras)\n\n"
            "Regular videos without gyro data cannot use this mode.\n"
            "Try OpenCV or RAFT modes instead."
        )
        return None

    emit('detect', 15, message="Gyroscope data found")

    # Create preset file if needed (Gyroflow CLI may support direct args)
    # For now, use simple command-line args

    emit('transform', 20, message="Starting Gyroflow stabilization")

    # Build Gyroflow command
    # Note: Actual CLI arguments depend on gyroflow-cli version
    # This is a simplified version - adjust based on actual CLI

    cmd = [
        gyroflow_path,
        input_path,
        '--output', output_path,
        '--smoothing', str(settings.get('smoothing', 0.5)),
    ]

    if settings.get('horizon_lock'):
        cmd.append('--horizon-lock')

    if settings.get('fov') != 0:
        cmd.extend(['--fov', str(1.0 + settings.get('fov', 0) / 100.0)])

    if settings.get('lens_profile') and settings.get('lens_profile') != 'auto':
        cmd.extend(['--lens', settings.get('lens_profile')])

    if settings.get('resolution') != 'source':
        cmd.extend(['--resolution', settings.get('resolution')])

    try:
        # Run Gyroflow CLI
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Parse progress from stdout
        for line in process.stdout:
            line = line.strip()

            # Try to parse progress (format varies by Gyroflow version)
            # Common formats: "Progress: 45%", "45.2%", "[45%]"

            if '%' in line:
                try:
                    # Extract percentage
                    percent_str = ''.join(c for c in line if c.isdigit() or c == '.')
                    if percent_str:
                        percent = float(percent_str)
                        # Map 0-100% to 20-95% overall progress
                        overall = 20 + (percent * 0.75)
                        emit('transform', overall, message=f"Stabilizing ({percent:.0f}%)")
                except ValueError:
                    pass

        # Wait for completion
        returncode = process.wait()

        if returncode != 0:
            stderr = process.stderr.read()
            emit_error(f"Gyroflow failed: {stderr}")
            return None

        emit('transform', 100, message="Stabilization complete")

        # Get output size
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        return {"ok": True, "outputSize": output_size}

    except Exception as e:
        emit_error(f"Gyroflow execution failed: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Gyroflow CLI integration")

    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output video file')
    parser.add_argument('--gyroflow', required=True, help='Path to gyroflow-cli')
    parser.add_argument('--ffprobe', required=True, help='Path to ffprobe')
    parser.add_argument('--smoothing', type=float, default=0.5,
                       help='Smoothing strength (0.0-1.0)')
    parser.add_argument('--fov', type=float, default=0,
                       help='Field of view adjustment (-10 to +10 percent)')
    parser.add_argument('--lens-profile', default='auto',
                       help='Lens correction profile (auto or specific profile)')
    parser.add_argument('--horizon-lock', action='store_true',
                       help='Enable horizon lock (level stabilization)')
    parser.add_argument('--resolution', default='source',
                       choices=['source', '1080p', '720p', '480p'],
                       help='Output resolution')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        emit_error(f"Input file not found: {args.input}")
        return 1

    if not os.path.exists(args.gyroflow):
        emit_error(f"Gyroflow CLI not found: {args.gyroflow}")
        return 1

    if not os.path.exists(args.ffprobe):
        emit_error(f"ffprobe not found: {args.ffprobe}")
        return 1

    # Gather settings
    settings = {
        'smoothing': args.smoothing,
        'fov': args.fov,
        'lens_profile': args.lens_profile,
        'horizon_lock': args.horizon_lock,
        'resolution': args.resolution,
    }

    # Run Gyroflow
    result = run_gyroflow(args.gyroflow, args.input, args.output, settings, args.ffprobe)

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
