# Video Stabbot

Professional video stabilization application with advanced motion analysis and AI-powered techniques.

## Description

Video Stabbot is a desktop video stabilization application built with Electron. It provides a polished dark-themed UI for stabilizing shaky video footage using multiple cutting-edge technologies: FFmpeg's vidstab filters, OpenCV feature tracking, Gyroflow gyroscope-based stabilization, and RAFT deep learning optical flow.

## Features

### FFmpeg Modes
- **High Quality Mode** — Maximum analysis depth and smoothing using FFmpeg vidstab for excellent results
- **Custom Mode** — Full manual control over all vidstab parameters:
  - Border crop mode, smoothing strength (1-300), shakiness detection
  - Accuracy, auto/manual zoom, zoom speed, interpolation method
  - Tripod mode, relative transforms, max shift, max rotation
  - Step size, minimum contrast, encoding quality

### Advanced Stabilization Modes

- **OpenCV Feature Tracking** (requires Python + SciPy) — Feature-based stabilization with advanced trajectory smoothing
  - SIFT/ORB/AKAZE feature detection algorithms
  - Affine or homography transform estimation
  - Multiple smoothing methods: Savitzky-Golay, Gaussian, Moving Average, Cubic Spline
  - Configurable smoothing strength, crop percentage, and resolution

- **Gyroflow Integration** (requires Gyroflow CLI) — Gyroscope-based stabilization for cameras with embedded motion data
  - Auto-detects gyroscope metadata (GoPro, DJI, Insta360, Sony)
  - Smoothing strength and FOV adjustment controls
  - Horizon lock for level stabilization
  - Lens profile selection for accurate correction
  - Best-in-class quality for supported cameras

- **RAFT Deep Learning** (requires PyTorch) — AI-powered dense optical flow for ultimate stabilization quality
  - Uses RAFT neural network for dense motion estimation
  - RAFT-Sintel (natural videos) or RAFT-Things (general) model variants
  - Configurable refinement iterations (6-20)
  - Advanced trajectory smoothing with multiple algorithms
  - GPU acceleration support (CUDA highly recommended)
  - Best quality but slowest processing

### General Features
- **GPU Acceleration** — Auto-detects NVIDIA NVENC, Intel QSV, AMD AMF, or Apple VideoToolbox; falls back to CPU (libx264)
- **Smart Dependency Detection** — Advanced modes automatically show/hide based on installed dependencies
- **Drag-and-drop or file-picker** — Easy video input
- **Real-time progress tracking** — Phase labels, progress bar, and elapsed time
- **Tooltips** — Hover over any setting for detailed explanations
- **Bidirectional slider/input controls** — Type values manually or use sliders

## Prerequisites

### Core Requirements
- **Node.js** (v18 or later) — https://nodejs.org
- **FFmpeg** with **libvidstab** support — must be on your system PATH
  - Download from https://ffmpeg.org/download.html
  - Verify: `ffmpeg -filters | findstr vidstab` (Windows) or `ffmpeg -filters | grep vidstab` (macOS/Linux)

### Optional: Advanced Modes

#### OpenCV Feature Tracking Mode
- **Python 3.8+** — on your system PATH
- **Required packages**:
  ```bash
  pip install opencv-python numpy scipy
  ```

#### Gyroflow Mode
- **Gyroflow CLI** — Download from https://gyroflow.xyz/download
  - Windows: Install to `C:\Program Files\Gyroflow\` or add to PATH
  - macOS: Install to `/Applications/Gyroflow.app/`
  - Linux: Install to `/usr/bin/gyroflow-cli` or add to PATH
- Compatible cameras with gyroscope data:
  - GoPro Hero 5 and newer
  - DJI drones and action cameras (Osmo Action, etc.)
  - Insta360 cameras (various models)
  - Sony cameras (select models with gyro metadata)

#### RAFT Deep Learning Mode
- **Python 3.8+** — on your system PATH
- **PyTorch and torchvision** (large install ~2-4GB):
  ```bash
  # CPU-only version
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

  # GPU version (NVIDIA CUDA) - highly recommended
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- **Additional packages**:
  ```bash
  pip install opencv-python numpy scipy
  ```
- **Note**: First run will auto-download the RAFT model (~250MB). GPU strongly recommended for acceptable performance.

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd Video-Stabbot

# Install Node dependencies
npm install

# Install Python dependencies (for advanced modes)
pip install -r requirements.txt
```

## Running

```bash
npm start
```

This launches the Electron app. From there:

1. **Select a video** — Drag onto the drop zone or click to browse
2. **Choose a stabilization mode**:
   - FFmpeg modes: High Quality or Custom
   - Advanced modes (if dependencies installed): OpenCV Features, Gyroflow, or RAFT Deep Learning
3. **Configure settings** — Each mode has its own settings view with tooltips
4. **Pick a save location** — Choose where to save the stabilized video
5. **Wait for processing** — Progress bar shows current phase and elapsed time
6. **View result** — Open output folder or process another video

## Project Structure

```
Video-Stabbot/
├── package.json              NPM project config
├── requirements.txt          Python dependencies for advanced modes
├── .gitignore
├── README.md                 This file
├── src/
│   ├── main/
│   │   ├── main.js           Electron main process (system detection, processing)
│   │   └── preload.js        Secure IPC bridge
│   └── renderer/
│       ├── index.html        Application UI (all views)
│       ├── renderer.js       Frontend logic (settings, progress, handlers)
│       └── styles.css        Dark theme with tooltips
└── scripts/
    ├── smoothing_lib.py      Shared trajectory smoothing library (SciPy-based)
    ├── opencv_feature_tracking.py    SIFT/ORB/AKAZE feature-based stabilization
    ├── gyroflow_integration.py       Gyroflow CLI wrapper
    └── raft_dense_motion.py          RAFT deep learning stabilization
```

## Mode Comparison

| Mode | Speed | Quality | Requirements | Best For |
|------|-------|---------|--------------|----------|
| **High Quality** | Fast | Good | FFmpeg only | General use, quick results |
| **Custom** | Fast | Good-Excellent | FFmpeg only | Fine-tuning parameters |
| **OpenCV Features** | Medium | Excellent | Python + SciPy | Complex motion, superior quality |
| **Gyroflow** | Fast-Medium | Best | Gyroflow CLI + gyro data | GoPro/DJI/Insta360 footage |
| **RAFT Deep Learning** | Slow | Ultimate | Python + PyTorch + GPU | Maximum quality (time permitting) |

## Troubleshooting

### Advanced modes don't show up
- **OpenCV**: Install `pip install scipy` - required for trajectory smoothing
- **Gyroflow**: Install Gyroflow CLI from https://gyroflow.xyz/download
- **RAFT**: Install `pip install torch torchvision` (large download ~2-4GB)

### RAFT mode is very slow
- RAFT requires significant computing power
- Install PyTorch with CUDA support for GPU acceleration
- Reduce refinement iterations (6 instead of 12) for faster processing
- Consider using OpenCV mode instead for CPU-only systems

### Gyroflow says "No gyroscope data detected"
- Not all cameras embed gyroscope metadata
- Check if your camera is supported: GoPro Hero 5+, DJI drones/action cams, Insta360
- Regular smartphone/camera videos without gyro data cannot use Gyroflow mode
- Try OpenCV or RAFT modes instead

### FFmpeg vidstab filters not found
- Download FFmpeg build with libvidstab support
- Verify: run `ffmpeg -filters | grep vidstab` in terminal
- Windows users: try builds from https://www.gyan.dev/ffmpeg/builds/

## Building a Distributable

To package the app as a standalone executable:

```bash
# Install electron-builder
npm install --save-dev electron-builder

# Build for your platform
npx electron-builder --win   # Windows
npx electron-builder --mac   # macOS
npx electron-builder --linux # Linux
```

The output will be in the `dist/` folder.

**Note**: Python dependencies (scipy, torch) must be installed separately by end users for advanced modes to work.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- FFmpeg vidstab filters for core stabilization
- OpenCV for feature detection and tracking
- Gyroflow for gyroscope-based stabilization
- RAFT (Recurrent All-Pairs Field Transforms) for optical flow
- SciPy for advanced trajectory smoothing algorithms
