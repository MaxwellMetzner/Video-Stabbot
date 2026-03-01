# Video Stabbot

Professional video stabilization application with advanced motion analysis and AI-powered techniques.

## Description

Video Stabbot is a desktop video stabilization application built with Electron. It provides a polished dark-themed UI for stabilizing shaky video footage using multiple cutting-edge technologies: FFmpeg's vidstab filters, OpenCV feature tracking, and RAFT deep learning optical flow.

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

- **RAFT Deep Learning** (requires PyTorch) — AI-powered dense optical flow for ultimate stabilization quality
  - Uses RAFT neural network for dense motion estimation
  - RAFT-Sintel (natural videos) or RAFT-Things (general) model variants
  - Configurable refinement iterations (6-20)
  - Advanced trajectory smoothing with multiple algorithms
  - GPU acceleration support (CUDA highly recommended)
  - Best quality but slowest processing

### General Features

- **GPU Acceleration** — Auto-detects NVIDIA NVENC, Intel QSV, AMD AMF, or Apple VideoToolbox; falls back to CPU (libx264)
- **Smart Dependency Detection** — Advanced modes appear grayed out with tooltip explanations when prerequisites are missing
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

#### RAFT Deep Learning Mode

- **Python 3.8+** — on your system PATH
- **PyTorch and torchvision** (large install ~2-4GB):

  ```bash
  # GPU version (NVIDIA CUDA) - highly recommended
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

  # CPU-only version
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
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
   - Advanced modes (if dependencies installed): OpenCV Features or RAFT Deep Learning
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
    └── raft_dense_motion.py          RAFT deep learning stabilization
```

## Mode Comparison

| Mode | Speed | Quality | Requirements | Best For |
| ------ | ------- | --------- | -------------- | ---------- |
| **High Quality** | Fast | Good | FFmpeg only | General use, quick results |
| **Custom** | Fast | Good-Excellent | FFmpeg only | Fine-tuning parameters |
| **OpenCV Features** | Medium | Excellent | Python + SciPy | Complex motion, superior quality |
| **RAFT Deep Learning** | Slow | Ultimate | Python + PyTorch + GPU | Maximum quality (time permitting) |

## Troubleshooting

### Advanced modes are grayed out

- **OpenCV**: Install `pip install scipy` — required for trajectory smoothing
- **RAFT**: Install `pip install torch torchvision opencv-python numpy scipy` — hover over the grayed-out button for specific missing packages

### RAFT mode is very slow

- RAFT requires significant computing power
- Install PyTorch with CUDA support for GPU acceleration
- Reduce refinement iterations (6 instead of 12) for faster processing
- Consider using OpenCV mode instead for CPU-only systems

### FFmpeg vidstab filters not found

- Download FFmpeg build with libvidstab support
- Verify: run `ffmpeg -filters | grep vidstab` in terminal
- Windows users: try builds from https://www.gyan.dev/ffmpeg/builds/

## Building a Distributable

To package the app as a standalone Windows executable (no console window):

```bash
# Build Windows release artifacts
npm run dist:win
```

Output files are created in `dist/`:

- `video-stabbot Setup <version>.exe` (installer)
- `video-stabbot <version>.exe` (portable single executable, if generated)

You can double-click either executable to launch the app directly.

Optional build commands:

```bash
# Build unpacked app folder (quick local packaging test)
npm run pack

# Build all configured targets
npm run dist
```

**Note**: Python dependencies (scipy, torch) must be installed separately by end users for advanced modes to work.

## GitHub + Releases Workflow

### What goes in GitHub (commit these)

Commit source and project metadata only:

- `src/`
- `scripts/`
- `package.json`
- `package-lock.json`
- `requirements.txt`
- `README.md`
- `.gitignore`
- Any icons/assets/config files you intentionally add for packaging

Do **not** commit generated build artifacts (`dist/`, unpacked app folders, installers, portable exes).

### What to upload under GitHub Releases

After running `npm run dist:win`, upload these files from `dist/` to the release:

- `video-stabbot Setup <version>.exe` (recommended for most users)
- `video-stabbot <version>.exe` (portable build, if generated)

Optional:

- `latest.yml` and `.blockmap` files only if you later add auto-update support

Do not upload `win-unpacked/` (developer/debug artifact, very large, not needed by end users).

### Recommended release steps (Windows)

1. Update version in `package.json` (example: `2.0.1`)
2. Commit and push changes:

  ```bash
  git add .
  git commit -m "Release v2.0.1"
  git push
  ```

3. Build release artifacts:

  ```bash
  npm run dist:win
  ```

4. Create and push tag:

  ```bash
  git tag v2.0.1
  git push origin v2.0.1
  ```

5. In GitHub: **Releases** → **Draft a new release**
  - Tag: `v2.0.1`
  - Title: `Video Stabbot v2.0.1`
  - Upload the `.exe` asset(s) from `dist/`
  - Publish release

### End-user install guidance

- Most users should download the `Setup` exe and install normally.
- Users who prefer no installation can use the portable exe.
- Advanced modes still require local Python + packages (`opencv-python`, `numpy`, `scipy`, and for RAFT: `torch`, `torchvision`).

## License

MIT License - see LICENSE file for details

## Acknowledgments

- FFmpeg vidstab filters for core stabilization
- OpenCV for feature detection and tracking
- Gyroflow for gyroscope-based stabilization
- RAFT (Recurrent All-Pairs Field Transforms) for optical flow
- SciPy for advanced trajectory smoothing algorithms
