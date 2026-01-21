# Video Stabbot

## Description

Video Stabbot is a Python script that automatically stabilizes videos. It utilizes FFmpeg with the `vidstabdetect` and `vidstabtransform` filters to analyze video motion and apply stabilization. The process involves detecting camera shake and then transforming the video to counteract this shake, typically by adjusting the frame within a black border to avoid losing content at the edges due to movement.

## How it Works

The script performs two main steps using FFmpeg:

1.  **Detection Phase**:
    *   Command: `ffmpeg -i <input_file> -vf vidstabdetect=result="transforms.trf" -f null -`
    *   This step analyzes the input video (`<input_file>`) to detect camera movements and shakes. The results of this analysis (motion vectors) are saved to a file named `transforms.trf` in the same directory as the script.

2.  **Transformation Phase**:
    *   Command: `ffmpeg -y -i <input_file> -vf vidstabtransform=input="transforms.trf":smoothing=30:crop=black:zoom=-15:optzoom=0:interpol=linear <output_file>`
    *   This step uses the motion data from `transforms.trf` to stabilize the video.
        *   `smoothing=30`: Adjusts the strength of the stabilization. Higher values mean smoother results but can lead to more cropping or "floating" effects.
        *   `crop=black`: Fills the areas exposed by stabilization with black.
        *   `zoom=-15`: Zooms out slightly to create a margin for stabilization movements. A negative value means zooming out.
        *   `optzoom=0`: Disables dynamic zooming.
        *   `interpol=linear`: Specifies the interpolation method.
    *   The stabilized video is saved to `<output_file>`.

After these steps, the temporary `transforms.trf` file is automatically deleted.

## Prerequisites

*   **Python 3**: The script is written in Python 3.
*   **FFmpeg**: You must have FFmpeg installed and accessible in your system's PATH. FFmpeg is a free and open-source software project consisting of a vast software suite of libraries and programs for handling video, audio, and other multimedia files and streams. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).

## Usage

To use the script, run it from your terminal:

```bash
python stabbot.py <input_video_path> <output_video_path>
```

Replace `<input_video_path>` with the path to the video file you want to stabilize, and `<output_video_path>` with the desired path for the stabilized output video.

**Example:**

```bash
python stabbot.py "my_shaky_video.mp4" "stabilized_video.mp4"
```

The script will then process the video and save the stabilized version to the specified output file. You'll see progress messages in the terminal for each step.