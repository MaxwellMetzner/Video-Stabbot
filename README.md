# Video Stabbot

Video Stabbot is a desktop app for stabilizing shaky video. Choose a quick automatic mode or adjust settings for more control.

## Stabilization modes

| Mode | Best for | Requirements |
| --- | --- | --- |
| **Quick FFmpeg** | Fast, general-purpose stabilization | FFmpeg with libvidstab |
| **Custom** | Tuning FFmpeg stabilization settings | FFmpeg with libvidstab |
| **OpenCV Features** | More involved handheld motion | Python, OpenCV, NumPy, SciPy |
| **RAFT Deep Learning** | Difficult motion when processing time is less important | Python, PyTorch, torchvision, OpenCV, NumPy, SciPy |

Quick FFmpeg uses accuracy 15 and step size 6, with smoothing scaled to about one second of video. OpenCV and RAFT use Gaussian motion smoothing by default and limit corrections to the selected crop.

## Requirements

- Node.js 18 or later
- FFmpeg with **libvidstab** support, available on your `PATH`
- Python 3.8 or later for OpenCV and RAFT modes

Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html). To check for libvidstab, run `ffmpeg -filters` and look for `vidstabdetect` and `vidstabtransform`.

The advanced modes need Python packages. Install them with:

```bash
pip install -r requirements.txt
```

This includes PyTorch and torchvision for RAFT; their installation can require several gigabytes. For NVIDIA GPU support, install the matching CUDA version of PyTorch by following the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Run the app

```bash
npm install
npm start
```

Choose a video, select a stabilization mode, adjust settings if needed, then choose where to save the result. The app displays processing progress and elapsed time.

## Troubleshooting

- **A mode is disabled:** Install its required Python packages. Hover over the mode for details.
- **FFmpeg filters are missing:** Install an FFmpeg build with libvidstab support and make sure FFmpeg is on your `PATH`.
- **RAFT is slow:** A supported GPU can speed up processing. For CPU-only systems, try OpenCV Features.

## Build for Windows

```bash
npm run dist:win
```

Install Python dependencies separately on computers using the advanced modes. Build outputs are created in `dist/`.

## License

MIT. See [LICENSE](./LICENSE).
