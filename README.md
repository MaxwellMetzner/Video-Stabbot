# Video Stabbot

Desktop video stabilization for quick fixes, manual FFmpeg tuning, OpenCV feature tracking, and RAFT dense optical flow.

## Description

Video Stabbot is an Electron app for stabilizing shaky footage from a single interface. It combines FFmpeg vidstab, OpenCV sparse optical-flow tracking, and RAFT dense optical flow so users can choose between speed, control, and maximum motion-analysis quality.

## Screenshots

![File selection](./screenshot-file-selection.png)
![Mode selection](./screenshot-mode-selection.png)
![Custom settings](./screenshot-custom-settings.png)
![Processing](./screenshot-processing.png)

## Features

### FFmpeg Modes

- **Quick FFmpeg Mode** - Fast two-pass stabilization using FFmpeg vidstab with bicubic transforms, recommended sharpening, optional audio mapping, and practical defaults.

- **Custom Mode** - Full manual control over vidstab parameters:
  - Border mode, smoothing strength, shakiness detection
  - Accuracy, auto/manual zoom, zoom speed, interpolation method
  - Tripod mode, relative transforms, max shift, max rotation
  - Step size, minimum contrast, encoding quality

### Advanced Stabilization Modes

- **OpenCV Feature Tracking** (requires Python, OpenCV, NumPy, and SciPy) - Sparse optical-flow stabilization with robust trajectory smoothing.
  - Auto Track mode uses Shi-Tomasi corners and pyramidal Lucas-Kanade tracking.
  - Features are distributed across a grid so motion estimates do not overfit one textured area.
  - SIFT, ORB, and AKAZE can seed tracking points when preferred.
  - Forward/backward track validation rejects unstable points.
  - ECC refinement improves the tracked transform when image alignment converges.
  - Farneback dense optical flow is used as a fallback when sparse tracks are weak.
  - RANSAC estimates camera motion while rejecting moving-object outliers.
  - Output keeps the selected resolution and muxes source audio.

- **RAFT Deep Learning** (requires PyTorch and torchvision) - Dense mesh-flow stabilization for hard footage.
  - Uses torchvision RAFT large weights for Sintel-style or Things-style motion.
  - Uses a 7 x 9 local mesh by default instead of only one global camera path.
  - Smooths local mesh trajectories to reduce parallax and rolling local jitter.
  - Automatically caps analysis resolution for practical memory use.
  - Converts dense flow to robust global camera motion with RANSAC.
  - Falls back to sparse tracking when dense flow is unreliable.
  - Output keeps the selected resolution and muxes source audio.

### General Features

- **Faster startup checks** - Python packages are probed in one lightweight pass instead of importing each dependency.
- **GPU acceleration** - Auto-detects NVIDIA NVENC, Intel QSV, AMD AMF, or Apple VideoToolbox; falls back to CPU libx264.
- **Smart dependency detection** - Advanced modes appear disabled with tooltip explanations when prerequisites are missing.
- **Drag-and-drop or file picker** - Easy video input.
- **Real-time progress tracking** - Phase labels, progress bar, and elapsed time.
- **Tooltips** - Hover over any setting for detailed explanations.
- **Bidirectional slider/input controls** - Type values manually or use sliders.

## Prerequisites

### Core Requirements

- **Node.js** v18 or later - https://nodejs.org
- **FFmpeg** with **libvidstab** support - must be on your system PATH
  - Download from https://ffmpeg.org/download.html
  - Verify: `ffmpeg -filters | findstr vidstab` on Windows or `ffmpeg -filters | grep vidstab` on macOS/Linux

### Optional: Advanced Modes

#### OpenCV Feature Tracking Mode

- **Python 3.8+** on your system PATH
- **Required packages**:

  ```bash
  pip install opencv-python numpy scipy
  ```

#### RAFT Deep Learning Mode

- **Python 3.8+** on your system PATH
- **PyTorch and torchvision**:

  ```bash
  # GPU version for NVIDIA CUDA
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

  # CPU-only version
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

- **Additional packages**:

  ```bash
  pip install opencv-python numpy scipy
  ```

First RAFT run may download model weights. GPU is strongly recommended.

## Setup

```bash
git clone <repo-url>
cd Video-Stabbot
npm install
pip install -r requirements.txt
```

## Running

```bash
npm start
```

From the app:

1. **Select a video** - Drag onto the drop zone or click to browse.
2. **Choose a stabilization mode** - Quick FFmpeg, Custom, OpenCV Features, or RAFT Deep Learning.
3. **Configure settings** - Each advanced mode has its own settings view with tooltips.
4. **Pick a save location** - Choose where to save the stabilized video.
5. **Wait for processing** - Progress shows the current phase and elapsed time.
6. **View result** - Open the output folder or process another video.

## Project Structure

```text
Video-Stabbot/
|-- package.json
|-- requirements.txt
|-- README.md
|-- screenshot-file-selection.png
|-- screenshot-mode-selection.png
|-- screenshot-custom-settings.png
|-- screenshot-processing.png
|-- src/
|   |-- main/
|   |   |-- main.js
|   |   |-- detect-worker.js
|   |   `-- preload.js
|   `-- renderer/
|       |-- index.html
|       |-- renderer.js
|       `-- styles.css
`-- scripts/
    |-- smoothing_lib.py
    |-- opencv_feature_tracking.py
    `-- raft_dense_motion.py
```

## Mode Comparison

| Mode | Speed | Quality | Requirements | Best For |
| --- | --- | --- | --- | --- |
| **Quick FFmpeg** | Fast | Good | FFmpeg only | General use and quick results |
| **Custom** | Fast to medium | Good to excellent | FFmpeg only | Fine-tuning vidstab parameters |
| **OpenCV Features** | Medium | Excellent | Python, OpenCV, NumPy, SciPy | Complex handheld motion and CPU-friendly quality |
| **RAFT Deep Learning** | Very slow | Highest | Python, PyTorch, torchvision, SciPy | Difficult footage where processing time is acceptable |

## Troubleshooting

### Advanced modes are disabled

- **OpenCV**: Install `pip install opencv-python numpy scipy`.
- **RAFT**: Install `pip install torch torchvision opencv-python numpy scipy`.
- Hover over the disabled mode tile for the specific missing dependency.

### RAFT mode is very slow

- Install PyTorch with CUDA support for GPU acceleration.
- Reduce refinement iterations only when speed matters more than quality.
- Use OpenCV mode on CPU-only systems when speed matters.

### FFmpeg vidstab filters not found

- Download an FFmpeg build with libvidstab support.
- Verify with `ffmpeg -filters | grep vidstab`.
- Windows users can try builds from https://www.gyan.dev/ffmpeg/builds/.

## Building a Distributable

To package the app as a standalone Windows executable:

```bash
npm run dist:win
```

Output files are created in `dist/`:

- `video-stabbot Setup <version>.exe`
- `video-stabbot <version>.exe` if the portable target is generated

Optional build commands:

```bash
npm run pack
npm run dist
```

Python dependencies must still be installed separately by end users for advanced modes.

## GitHub and Releases Workflow

Commit source and project metadata only:

- `src/`
- `scripts/`
- `package.json`
- `package-lock.json`
- `requirements.txt`
- `README.md`
- screenshot pngs or other intentional assets
- `.gitignore`

Do not commit generated build artifacts such as `dist/`, installers, portable executables, or unpacked app folders.

After running `npm run dist:win`, upload release artifacts from `dist/`:

- `video-stabbot Setup <version>.exe`
- `video-stabbot <version>.exe` if generated

Recommended release steps on Windows:

1. Update version in `package.json`.
2. Commit and push changes.
3. Build release artifacts with `npm run dist:win`.
4. Create and push a tag such as `v2.0.1`.
5. Draft a GitHub release for that tag and upload the `.exe` assets.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- FFmpeg vidstab filters for the quick and custom stabilization paths.
- OpenCV for feature detection, Lucas-Kanade optical flow, and robust affine estimation.
- RAFT for dense optical flow.
- SciPy for trajectory smoothing.
