# asr-tools

## Usage

The script will produce a VTT of a single video file in mp4 format. Run via the command line with [uv](https://docs.astral.sh/uv/):

```bash
uv run inference.py my_file.mp4
```

Unless you are running this on a fairly powerful system, the basic usage will probably cause either a segmentation fault or a CUDA out of memory issue. To prevent this, the script comes with a flag that reduces the attention model and overall CUDA demands:

```bash
uv run inference.py --low-attention my_file.mp4
```

The script outputs a .vtt file with the same base name in the same directory the file is currently in.

## Installation

uv should handle installation of the python dependencies.

You may run into issues if you do not have packages installed to build python extensions. On Debian/Ubuntu, this can be resolved through:

```bash
sudo apt-get install build-essential -y
sudo apt-get install python3-dev
```
