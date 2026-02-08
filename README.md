# Video Stabbot

TODO: The panorama and mosaic modes need to be fixed

## Description

Video Stabbot is a desktop video stabilization application built with Electron. It provides a polished dark-themed UI for stabilizing shaky video footage using FFmpeg's vidstab filters, with optional Python+OpenCV modes for subject-locked reframing and panorama mosaic creation.

## Features

- **Quick Mode** — Fast stabilization with sensible defaults
- **High Quality Mode** — Maximum analysis depth and smoothing for the best result
- **Custom Mode** — Full control over all vidstab parameters:
  - Border mode, smoothing, shakiness, accuracy
  - Auto/manual zoom, interpolation method
  - Tripod mode, max shift, max rotation
  - Step size, minimum contrast, encoding quality
- **Subject Lock** (requires Python + OpenCV) — Tracks a face or point and reframes the video to keep the subject centred
- **Panorama / Mosaic** (requires Python + OpenCV) — Accumulates frames into a wide panoramic image or stabilized panorama video using feature matching and homography
- **GPU acceleration** — Auto-detects NVIDIA NVENC, Intel QSV, AMD AMF, or Apple VideoToolbox; falls back to CPU (libx264)
- Drag-and-drop or file-picker input
- Real-time progress bar with phase labels and elapsed time

## Prerequisites

- **Node.js** (v18 or later) — https://nodejs.org
- **FFmpeg** with **libvidstab** support — must be on your system PATH
  - Download from https://ffmpeg.org/download.html
  - Verify: `ffmpeg -filters | findstr vidstab` (Windows) or `ffmpeg -filters | grep vidstab` (macOS/Linux)

### Optional (for Subject Lock & Panorama modes)

- **Python 3.8+** — on your system PATH
- **opencv-python** and **numpy**:
  ```
  pip install opencv-python numpy
  ```

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd Video-Stabbot

# Install Node dependencies
npm install
```

## Running

```bash
npm start
```

This launches the Electron app. From there:

1. Drag a video onto the drop zone or click to browse.
2. Choose a stabilization mode (Quick, High Quality, Custom, Subject Lock, or Panorama).
3. Pick a save location.
4. Wait for processing to complete — the progress bar shows the current phase and elapsed time.
5. Open the output folder or start another video.

## Project Structure

```
Video-Stabbot/
├── package.json              NPM project config
├── requirements.txt          Python dependencies for advanced modes
├── .gitignore
├── src/
│   ├── main/
│   │   ├── main.js           Electron main process (system detection, FFmpeg/Python processing)
│   │   └── preload.js        Secure IPC bridge between main and renderer
│   └── renderer/
│       ├── index.html        Application UI (all views)
│       ├── renderer.js       Frontend logic (view management, settings, progress)
│       └── styles.css        Dark theme styling
└── scripts/
    ├── reframe.py            Subject-locked reframing (face/point tracking)
    └── mosaic.py             Panorama/mosaic accumulation (ORB/SIFT features)
```

## Building a Distributable (optional)

To package the app as a standalone executable you can use [electron-builder](https://www.electron.build/) or [electron-forge](https://www.electronforge.io/):

```bash
# Example with electron-builder
npm install --save-dev electron-builder
npx electron-builder --win   # or --mac / --linux
```

The output will be in the `dist/` folder.
