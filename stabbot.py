"""
Video Stabbot - Automatic Video Stabilization Tool
Uses FFmpeg with vidstab filters to detect and correct camera shake.
"""

import sys
import subprocess
import os
import shutil
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tkinter as tk
from tkinter import filedialog, messagebox

# TODO: Check the output, glitchy in some circumstances, I think we have strayed too far from the original stabbot parameters.

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Configuration settings for video stabilization."""
    
    # Video format support
    SUPPORTED_FORMATS: set = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    # Stabilization presets
    STABILIZATION_PRESETS: Dict[str, Dict] = {
        "fast": {"smoothing": 10, "crop": "black", "zoom": -5, "shakiness": 5},
        "balanced": {"smoothing": 30, "crop": "black", "zoom": -15, "shakiness": 5},
        "high_quality": {"smoothing": 50, "crop": "black", "zoom": -25, "shakiness": 10},
    }
    
    # Encoder configurations
    ENCODERS: Dict = {
        "nvidia": (["h264_nvenc"], "nvenc"),
        "intel": (["h264_qsv"], "qsv"),
        "amd": (["h264_amf"], "amf"),
        "apple": (["h264_videotoolbox"], "hardware"),
    }
    
    # Quality presets for software encoding
    SOFTWARE_QUALITY: Dict[str, List] = {
        "high": ["-crf", "18", "-preset", "slower"],
        "balanced": ["-crf", "23", "-preset", "medium"],
        "compressed": ["-crf", "28", "-preset", "fast"],
    }
    
    # Quality presets for NVENC
    NVENC_QUALITY: Dict[str, List] = {
        "high": ["-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M"],
        "balanced": ["-b:v", "5M", "-maxrate", "8M", "-bufsize", "10M"],
        "compressed": ["-b:v", "3M", "-maxrate", "5M", "-bufsize", "6M"],
    }
    
    # Quality presets for QSV/AMF
    QSV_AMF_QUALITY: Dict[str, List] = {
        "high": ["-global_quality", "18"],
        "balanced": ["-global_quality", "23"],
        "compressed": ["-global_quality", "28"],
    }
    
    # Default settings
    DEFAULT_ENCODER = "libx264"
    DEFAULT_QUALITY = "balanced"
    DEFAULT_STABILIZATION = "balanced"
    DEFAULT_AUDIO_BITRATE = "128k"
    
    # Config file location
    CONFIG_FILE = Path.home() / ".stabbot_config.json"


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging to both console and optional file.
    
    Args:
        log_file: Optional path to log file. If None, only logs to console.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("stabbot")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class StabbotError(Exception):
    """Base exception for Stabbot errors."""
    pass


class FFmpegNotFoundError(StabbotError):
    """Raised when FFmpeg is not found in PATH."""
    pass


class FFmpegVidstabError(StabbotError):
    """Raised when vidstab filter is not available in FFmpeg."""
    pass


class ProcessingError(StabbotError):
    """Raised when video processing fails."""
    pass


class ValidationError(StabbotError):
    """Raised when input validation fails."""
    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def pause_on_error() -> None:
    """Pause to allow user to read error messages when run from File Explorer."""
    try:
        input("Press Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        pass


def find_ffmpeg() -> str:
    """
    Find FFmpeg executable in system PATH.
    
    Returns:
        Path to FFmpeg executable.
        
    Raises:
        FFmpegNotFoundError: If FFmpeg cannot be found.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.error("FFmpeg not found in PATH")
        logger.error("Please install FFmpeg from https://ffmpeg.org/download.html")
        raise FFmpegNotFoundError("FFmpeg not found in system PATH")
    return ffmpeg_path


def verify_vidstab_support(ffmpeg_path: str) -> bool:
    """
    Verify that FFmpeg has vidstab filter support.
    
    Args:
        ffmpeg_path: Path to FFmpeg executable.
        
    Returns:
        True if vidstab is supported, False otherwise.
        
    Raises:
        FFmpegVidstabError: If vidstab filter is not available.
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-filters"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "vidstabdetect" not in result.stdout or "vidstabtransform" not in result.stdout:
            logger.error("FFmpeg does not have vidstab filter support")
            logger.error("Please install FFmpeg with libvidstab support")
            raise FFmpegVidstabError(
                "vidstab filters not found in FFmpeg. "
                "Install FFmpeg with libvidstab support."
            )
        return True
        
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg filter check timed out, assuming vidstab available")
        return True
    except Exception as e:
        logger.warning(f"Could not verify vidstab support: {e}")
        return True

def detect_hardware_acceleration(ffmpeg_path: str) -> List[Tuple[str, str]]:
    """
    Detect available hardware acceleration options.
    
    Args:
        ffmpeg_path: Path to FFmpeg executable.
        
    Returns:
        List of (display_name, encoder_name) tuples for available encoders.
    """
    encoders: List[Tuple[str, str]] = []
    try:
        result = subprocess.run(
            [ffmpeg_path, "-encoders"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout
        
        encoder_map = {
            "h264_nvenc": "NVIDIA NVENC",
            "h264_qsv": "Intel Quick Sync",
            "h264_amf": "AMD AMF",
            "h264_videotoolbox": "Apple VideoToolbox",
        }
        
        for encoder, name in encoder_map.items():
            if encoder in output:
                encoders.append((name, encoder))
                logger.debug(f"Detected {name} encoder")
                
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.debug(f"Error detecting hardware acceleration: {e}")
    
    return encoders


def get_cpu_count() -> int:
    """Get number of CPU cores for threading."""
    return os.cpu_count() or 4


def delete_file(file_path: Path) -> None:
    """
    Delete a file safely.
    
    Args:
        file_path: Path to file to delete.
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted temporary file: {file_path}")
    except OSError as e:
        logger.warning(f"Could not delete file {file_path}: {e}")


def validate_input_file(input_path: Path) -> None:
    """
    Validate that input file exists and is a supported format.
    
    Args:
        input_path: Path to input file.
        
    Raises:
        ValidationError: If file doesn't exist or format not supported.
    """
    if not input_path.exists():
        raise ValidationError(f"Input file not found: {input_path}")
    
    if input_path.suffix.lower() not in Config.SUPPORTED_FORMATS:
        logger.warning(f"File format {input_path.suffix} may not be supported")


def validate_output_file(output_path: Path) -> None:
    """
    Validate output file path and check for existing file.
    
    Args:
        output_path: Path to output file.
        
    Raises:
        ValidationError: If output directory doesn't exist or is not writable.
    """
    output_dir = output_path.parent
    
    if not output_dir.exists():
        raise ValidationError(f"Output directory does not exist: {output_dir}")
    
    if not os.access(output_dir, os.W_OK):
        raise ValidationError(f"Output directory is not writable: {output_dir}")
    
    if output_path.exists():
        logger.warning(f"Output file already exists and will be overwritten: {output_path}")

def get_encoder_settings(available_encoders: List[Tuple[str, str]]) -> Tuple[str, List[str], str]:
    """
    Allow user to choose encoder for better performance.
    
    Args:
        available_encoders: List of available hardware encoders.
        
    Returns:
        Tuple of (encoder_name, encoder_flags, encoder_type).
    """
    if not available_encoders:
        logger.info("No hardware acceleration detected. Using software encoding.")
        return Config.DEFAULT_ENCODER, [], "software"
    
    print("\n[GPU] Hardware Acceleration Detected:")
    print("0. Software encoding (libx264) - slower but compatible")
    for i, (name, encoder) in enumerate(available_encoders, 1):
        print(f"{i}. {name} ({encoder}) - faster")
    
    while True:
        try:
            choice = input(f"\nChoose encoder (0-{len(available_encoders)}) [default: 0]: ").strip()
            if not choice:
                choice = "0"
            
            choice_idx = int(choice)
            if choice_idx == 0:
                logger.info("Using software encoder: libx264")
                return Config.DEFAULT_ENCODER, [], "software"
            elif 1 <= choice_idx <= len(available_encoders):
                encoder = available_encoders[choice_idx - 1][1]
                name = available_encoders[choice_idx - 1][0]
                logger.info(f"Using hardware encoder: {name} ({encoder})")
                
                if "nvenc" in encoder:
                    return encoder, ["-preset", "p4", "-rc", "vbr"], "nvenc"
                elif "qsv" in encoder:
                    return encoder, ["-preset", "fast"], "qsv"
                elif "amf" in encoder:
                    return encoder, ["-usage", "transcoding", "-rc", "vbr"], "amf"
                else:
                    return encoder, [], "hardware"
            else:
                print(f"Invalid choice. Please enter 0-{len(available_encoders)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            logger.info("Using default software encoder due to user interruption")
            return Config.DEFAULT_ENCODER, [], "software"


def get_quality_settings(encoder_type: str) -> List[str]:
    """
    Get quality settings based on encoder type.
    
    Args:
        encoder_type: Type of encoder (software, nvenc, qsv, amf, hardware).
        
    Returns:
        List of FFmpeg quality flags.
    """
    print("\n[*] Quality Settings:")
    print("1. High quality (larger file, slower encoding)")
    print("2. Balanced (good quality/size balance)")
    print("3. Compressed (smaller file, faster encoding)")
    
    quality_map = {"1": "high", "2": "balanced", "3": "compressed"}
    
    while True:
        try:
            choice = input(f"Choose quality (1-3) [default: 2]: ").strip()
            if not choice:
                choice = "2"
            
            if choice not in quality_map:
                print("Invalid choice. Please enter 1, 2, or 3.")
                continue
            
            quality = quality_map[choice]
            logger.info(f"Using quality preset: {quality}")
            
            if encoder_type == "software":
                return Config.SOFTWARE_QUALITY[quality]
            elif encoder_type == "nvenc":
                return Config.NVENC_QUALITY[quality]
            elif encoder_type in ["qsv", "amf"]:
                return Config.QSV_AMF_QUALITY[quality]
            else:
                # Fallback for other hardware encoders
                bitrate_map = {"high": "8M", "balanced": "5M", "compressed": "3M"}
                return ["-b:v", bitrate_map[quality]]
                
        except (EOFError, KeyboardInterrupt):
            logger.info("Using default balanced quality due to user interruption")
            return Config.SOFTWARE_QUALITY["balanced"] if encoder_type == "software" else ["-b:v", "5M"]


def get_stabilization_settings() -> Dict:
    """
    Get user preferences for stabilization quality vs speed.
    
    Returns:
        Dictionary with stabilization parameters.
    """
    print("\n[*] Stabilization Settings:")
    print("1. Fast (lower quality, faster processing)")
    print("2. Balanced (good quality, moderate speed)")
    print("3. High Quality (best quality, slower processing)")
    print("4. Custom (specify your own settings)")
    
    while True:
        try:
            choice = input("Choose stabilization mode (1-4) [default: 2]: ").strip()
            if not choice:
                choice = "2"
            
            if choice == "1":
                preset = Config.STABILIZATION_PRESETS["fast"]
                logger.info("Using fast stabilization preset")
                return preset
            elif choice == "2":
                preset = Config.STABILIZATION_PRESETS["balanced"]
                logger.info("Using balanced stabilization preset")
                return preset
            elif choice == "3":
                preset = Config.STABILIZATION_PRESETS["high_quality"]
                logger.info("Using high quality stabilization preset")
                return preset
            elif choice == "4":
                print("\nCustom Stabilization Settings:")
                smoothing = int(input("Smoothing (1-1000) [30]: ") or "30")
                crop = input("Crop mode (black/keep) [black]: ") or "black"
                zoom = int(input("Zoom percentage (-50 to 50) [-15]: ") or "-15")
                shakiness = int(input("Shakiness detection (1-10) [5]: ") or "5")
                logger.info(f"Using custom settings: smoothing={smoothing}, crop={crop}, zoom={zoom}, shakiness={shakiness}")
                return {"smoothing": smoothing, "crop": crop, "zoom": zoom, "shakiness": shakiness}
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
        except (EOFError, KeyboardInterrupt):
            logger.info("Using default balanced stabilization due to user interruption")
            return Config.STABILIZATION_PRESETS["balanced"]


def load_config() -> Dict:
    """
    Load user configuration from file.
    
    Returns:
        Configuration dictionary, or empty dict if file doesn't exist.
    """
    if Config.CONFIG_FILE.exists():
        try:
            with open(Config.CONFIG_FILE, 'r') as f:
                config = json.load(f)
                logger.debug(f"Loaded configuration from {Config.CONFIG_FILE}")
                return config
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    return {}




# ============================================================================
# FFMPEG COMMAND EXECUTION
# ============================================================================

def run_ffmpeg(command_list: List[str]) -> None:
    """
    Run FFmpeg command with basic error handling.
    
    Args:
        command_list: List of command arguments.
        
    Raises:
        ProcessingError: If FFmpeg command fails.
    """
    try:
        logger.debug(f"Running: {' '.join(command_list)}")
        subprocess.run(command_list, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        raise ProcessingError(f"FFmpeg command failed: {' '.join(command_list)}")
    except FileNotFoundError as e:
        logger.error(f"FFmpeg not found: {e}")
        raise ProcessingError("FFmpeg command not found")


def run_ffmpeg_with_progress(command_list: List[str], step_name: str) -> Tuple[float, float]:
    """
    Run FFmpeg command with progress animation.
    
    Args:
        command_list: List of command arguments.
        step_name: Name of the processing step for display.
        
    Returns:
        Tuple of (start_time, end_time) for performance metrics.
        
    Raises:
        ProcessingError: If FFmpeg command fails.
    """
    import threading
    
    stop_animation = threading.Event()
    start_time = time.time()
    
    def animate_progress() -> None:
        """Animate progress spinner."""
        chars = "|/-\\"
        i = 0
        while not stop_animation.is_set():
            print(f"\r{step_name} {chars[i % len(chars)]}", end="", flush=True)
            i += 1
            time.sleep(0.1)
    
    progress_thread = threading.Thread(target=animate_progress, daemon=True)
    progress_thread.start()
    
    try:
        logger.debug(f"Running: {' '.join(command_list)}")
        subprocess.run(command_list, check=True, capture_output=True, text=True)
        
        stop_animation.set()
        progress_thread.join(timeout=1)
        end_time = time.time()
        print(f"\r{step_name} ✓" + " " * 10)
        
        return (start_time, end_time)
        
    except subprocess.CalledProcessError as e:
        stop_animation.set()
        progress_thread.join(timeout=1)
        print(f"\r{step_name} " + " " * 10)
        logger.error(f"FFmpeg command failed")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        raise ProcessingError(f"FFmpeg command failed: {' '.join(command_list)}")
    except FileNotFoundError as e:
        stop_animation.set()
        progress_thread.join(timeout=1)
        print(f"\r{step_name} " + " " * 10)
        logger.error(f"FFmpeg not found: {e}")
        raise ProcessingError("FFmpeg command not found")


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def build_detection_command(
    ffmpeg_path: str,
    input_file: Path,
    transforms_file: Path,
    shakiness: int
) -> List[str]:
    """
    Build FFmpeg command for motion detection.
    
    Args:
        ffmpeg_path: Path to FFmpeg executable.
        input_file: Input video file path.
        transforms_file: Output transforms file path.
        shakiness: Shakiness detection level (1-10).
        
    Returns:
        List of command arguments.
    """
    threading_flags = ["-threads", str(get_cpu_count())]
    
    # Convert path to use forward slashes for FFmpeg compatibility
    transforms_path_str = str(transforms_file).replace("\\", "/")
    
    return [
        ffmpeg_path,
        "-i", str(input_file),
        "-vf", f"vidstabdetect=shakiness={shakiness}:result={transforms_path_str}",
        "-f", "null", "-"
    ] + threading_flags


def build_transform_command(
    ffmpeg_path: str,
    input_file: Path,
    output_file: Path,
    transforms_file: Path,
    encoder: str,
    encoder_flags: List[str],
    quality_flags: List[str],
    stab_settings: Dict
) -> List[str]:
    """
    Build FFmpeg command for video stabilization transformation.
    
    Args:
        ffmpeg_path: Path to FFmpeg executable.
        input_file: Input video file path.
        output_file: Output video file path.
        transforms_file: Transforms file from detection phase.
        encoder: Video encoder to use.
        encoder_flags: Encoder-specific flags.
        quality_flags: Quality-specific flags.
        stab_settings: Stabilization settings dictionary.
        
    Returns:
        List of command arguments.
    """
    threading_flags = ["-threads", str(get_cpu_count())]
    
    # Convert path to use forward slashes for FFmpeg compatibility
    transforms_path_str = str(transforms_file).replace("\\", "/")
    
    # Build stabilization filter
    if stab_settings["crop"] == "keep":
        filter_str = (
            f"vidstabtransform=smoothing={stab_settings['smoothing']}:crop=black:zoom={stab_settings['zoom']}:"
            f"optzoom=0:interpol=linear:input={transforms_path_str},"
            f"pad=iw:ih:(ow-iw)/2:(oh-ih)/2:black"
        )
    else:
        filter_str = (
            f"vidstabtransform=smoothing={stab_settings['smoothing']}:crop=black:zoom={stab_settings['zoom']}:"
            f"optzoom=0:interpol=linear:input={transforms_path_str}"
        )
    
    return (
        [
            ffmpeg_path,
            "-y",
            "-i", str(input_file)
        ]
        + threading_flags
        + [
            "-vf", filter_str,
            "-c:v", encoder,
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts"
        ]
        + encoder_flags
        + quality_flags
        + [
            "-c:a", "aac",
            "-b:a", Config.DEFAULT_AUDIO_BITRATE,
            str(output_file)
        ]
    )


def process_video(
    input_file: Path,
    output_file: Path,
    ffmpeg_path: str,
    encoder: str,
    encoder_flags: List[str],
    quality_flags: List[str],
    stab_settings: Dict
) -> Tuple[float, float, float]:
    """
    Process a single video file with stabilization.
    
    Args:
        input_file: Input video file path.
        output_file: Output video file path.
        ffmpeg_path: Path to FFmpeg executable.
        encoder: Video encoder to use.
        encoder_flags: Encoder-specific flags.
        quality_flags: Quality-specific flags.
        stab_settings: Stabilization settings dictionary.
        
    Returns:
        Tuple of (detection_time, transform_time, total_time) in seconds.
        
    Raises:
        ValidationError: If input/output validation fails.
        ProcessingError: If FFmpeg processing fails.
    """
    # Validate input and output files
    validate_input_file(input_file)
    validate_output_file(output_file)
    
    # Create temporary directory for transforms file
    with tempfile.TemporaryDirectory() as temp_dir:
        transforms_file = Path(temp_dir) / "transforms.trf"
        
        logger.info(f"Processing: {input_file.name}")
        logger.info(f"Output: {output_file.name}")
        logger.info(f"Using encoder: {encoder}")
        
        # Detection phase
        detect_cmd = build_detection_command(ffmpeg_path, input_file, transforms_file, stab_settings["shakiness"])
        print(f"\n🔍 Analyzing video for motion ({input_file.name})...")
        detect_start, detect_end = run_ffmpeg_with_progress(detect_cmd, "🔍 Detecting motion")
        detection_time = detect_end - detect_start
        logger.info(f"Motion detection completed in {detection_time:.1f}s")
        
        # Transformation phase
        transform_cmd = build_transform_command(
            ffmpeg_path, input_file, output_file, transforms_file,
            encoder, encoder_flags, quality_flags, stab_settings
        )
        print(f" Stabilizing video...")
        transform_start, transform_end = run_ffmpeg_with_progress(transform_cmd, "🎯 Applying stabilization")
        transform_time = transform_end - transform_start
        logger.info(f"Video stabilization completed in {transform_time:.1f}s")
        
        # Transforms file is automatically deleted when temp dir is cleaned up
    
    total_time = time.time() - (detect_start)
    return (detection_time, transform_time, total_time)


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def select_files_for_batch() -> List[Path]:
    """
    Allow user to select multiple files for batch processing.
    
    Returns:
        List of selected file paths.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        root.update()
        
        files = filedialog.askopenfilenames(
            title="Select Video Files for Batch Processing",
            filetypes=(
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("Video files", "*.mkv *.wmv *.flv *.webm *.m4v"),
                ("All files", "*.*")
            )
        )
        
        root.destroy()
        return [Path(f) for f in files]
    except Exception as e:
        logger.error(f"Error selecting batch files: {e}")
        return []


def select_output_directory() -> Optional[Path]:
    """
    Allow user to select output directory for batch processing.
    
    Returns:
        Selected directory path or None if cancelled.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        root.update()
        
        directory = filedialog.askdirectory(title="Select Output Directory for Stabilized Videos")
        
        root.destroy()
        
        return Path(directory) if directory else None
    except Exception as e:
        logger.error(f"Error selecting output directory: {e}")
        return None


# ============================================================================
# MAIN PROCESSING FLOW
# ============================================================================

def select_single_file() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Allow user to select input and output files for single processing.
    
    Returns:
        Tuple of (input_file, output_file) or (None, None) if cancelled.
    """
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.update()

        # Prompt for input file
        input_file = filedialog.askopenfilename(
            title="Select Input Video File",
            filetypes=(
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All files", "*.*")
            )
        )
        
        if not input_file:
            if root:
                root.destroy()
            return None, None

        # Prompt for output file
        output_file = filedialog.asksaveasfilename(
            title="Save Stabilized Video As...",
            defaultextension=".mp4",
            filetypes=(
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All files", "*.*")
            )
        )
        
        if root:
            root.destroy()
        
        if not output_file:
            return None, None
        
        return Path(input_file), Path(output_file)
    except Exception as e:
        logger.error(f"Error selecting files: {e}")
        if root:
            try:
                root.destroy()
            except:
                pass
        return None, None


def get_mode_selection() -> str:
    """
    Ask user whether to process single file or batch.
    
    Returns:
        "single" or "batch".
    """
    print("\n Processing Mode:")
    print("1. Single file")
    print("2. Batch processing (multiple files)")
    
    while True:
        choice = input("Choose mode (1-2) [default: 1]: ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            return "single"
        elif choice == "2":
            return "batch"
        else:
            print("Invalid choice. Please enter 1 or 2.")


def main_single_file(
    ffmpeg_path: str,
    available_encoders: List[Tuple[str, str]],
    log_file: Optional[Path] = None
) -> None:
    """
    Process a single video file.
    
    Args:
        ffmpeg_path: Path to FFmpeg executable.
        available_encoders: List of available hardware encoders.
        log_file: Optional path to log file.
    """
    input_file, output_file = select_single_file()
    
    if not input_file or not output_file:
        logger.info("No files selected. Exiting.")
        return
    
    # Get user settings
    encoder, encoder_flags, encoder_type = get_encoder_settings(available_encoders)
    quality_flags = get_quality_settings(encoder_type)
    stab_settings = get_stabilization_settings()
    
    try:
        print(f"\n Video Stabilization Starting")
        print(f" Input: {input_file.name}")
        print(f" Output: {output_file.name}")
        print(f"  Encoder: {encoder} with {get_cpu_count()} threads")
        
        detection_time, transform_time, total_time = process_video(
            input_file, output_file, ffmpeg_path,
            encoder, encoder_flags, quality_flags, stab_settings
        )
        
        logger.info(f" Stabilized video saved to {output_file}")
        print(f"\n Processing complete!")
        print(f"   Detection: {detection_time:.1f}s | Transformation: {transform_time:.1f}s | Total: {total_time:.1f}s")
        
    except (ValidationError, ProcessingError) as e:
        logger.error(f"Error processing video: {e}")
        print(f"\n Error: {e}")
        pause_on_error()
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n Unexpected error: {e}")
        pause_on_error()


def main_batch(
    ffmpeg_path: str,
    available_encoders: List[Tuple[str, str]],
    log_file: Optional[Path] = None
) -> None:
    """
    Process multiple video files in batch.
    
    Args:
        ffmpeg_path: Path to FFmpeg executable.
        available_encoders: List of available hardware encoders.
        log_file: Optional path to log file.
    """
    input_files = select_files_for_batch()
    
    if not input_files:
        logger.info("No files selected. Exiting.")
        return
    
    output_dir = select_output_directory()
    
    if not output_dir:
        logger.info("No output directory selected. Exiting.")
        return
    
    logger.info(f"Selected {len(input_files)} files for batch processing")
    
    # Get user settings (reused for all files)
    encoder, encoder_flags, encoder_type = get_encoder_settings(available_encoders)
    quality_flags = get_quality_settings(encoder_type)
    stab_settings = get_stabilization_settings()
    
    print(f"\n Batch Video Stabilization Starting")
    print(f" Processing {len(input_files)} files")
    print(f" Output directory: {output_dir}")
    print(f"  Encoder: {encoder} with {get_cpu_count()} threads\n")
    
    successful = 0
    failed = 0
    total_time = 0
    
    for idx, input_file in enumerate(input_files, 1):
        try:
            output_file = output_dir / f"{input_file.stem}_stabilized{input_file.suffix}"
            
            print(f"[{idx}/{len(input_files)}] Processing {input_file.name}...")
            
            detection_time, transform_time, proc_time = process_video(
                input_file, output_file, ffmpeg_path,
                encoder, encoder_flags, quality_flags, stab_settings
            )
            
            total_time += proc_time
            successful += 1
            logger.info(f" File {idx}/{len(input_files)}: {input_file.name} completed in {proc_time:.1f}s")
            
        except (ValidationError, ProcessingError) as e:
            failed += 1
            logger.error(f" File {idx}/{len(input_files)}: {input_file.name} failed - {e}")
            print(f"    Error: {e}")
    
    print(f"\n Batch Processing Complete!")
    print(f"    Successful: {successful}/{len(input_files)}")
    print(f"    Failed: {failed}/{len(input_files)}")
    print(f"     Total time: {total_time:.1f}s")
    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    """Main entry point for Stabbot."""
    try:
        # Set working directory to script location
        script_dir = Path(__file__).parent.resolve()
        os.chdir(script_dir)
        
        # Set up logging with log file
        log_file = script_dir / "stabbot.log"
        global logger
        
        # Add file handler to existing logger
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        logger.info("=" * 60)
        logger.info("Video Stabbot - Automatic Video Stabilization Tool")
        logger.info("=" * 60)
        
        # Find and validate FFmpeg
        try:
            ffmpeg_path = find_ffmpeg()
            logger.info(f"FFmpeg found at: {ffmpeg_path}")
        except FFmpegNotFoundError as e:
            print(f" {e}")
            pause_on_error()
            sys.exit(1)
        
        # Verify vidstab support
        try:
            verify_vidstab_support(ffmpeg_path)
        except FFmpegVidstabError as e:
            print(f" {e}")
            pause_on_error()
            sys.exit(1)
        
        # Detect hardware acceleration
        available_encoders = detect_hardware_acceleration(ffmpeg_path)
        
        # Get processing mode
        mode = get_mode_selection()
        
        # Process based on mode
        if mode == "single":
            main_single_file(ffmpeg_path, available_encoders, log_file)
        else:
            main_batch(ffmpeg_path, available_encoders, log_file)
        
        print("\n Thank you for using Video Stabbot!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\n\n  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n Fatal error: {e}")
        pause_on_error()
        sys.exit(1)




if __name__ == "__main__":
    main()
