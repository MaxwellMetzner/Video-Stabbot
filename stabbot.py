import sys
import subprocess
import os

def run_ffmpeg(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def delete_file(file):
    try:
        os.remove(file)
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"Error deleting file {file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 stabbot.py <input file> <output file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    detect_command = f"ffmpeg -i {input_file} -vf vidstabdetect -f null -"
    transform_command = f"ffmpeg -y -i {input_file} -vf vidstabtransform=smoothing=30:crop=black:zoom=-15:optzoom=0:interpol=linear {output_file}"
    delete_files_command = f"rm transforms.trf"

    run_ffmpeg(detect_command)
    run_ffmpeg(transform_command)
    run_ffmpeg(delete_files_command)

    print(f"Stabilized video saved to {output_file}")