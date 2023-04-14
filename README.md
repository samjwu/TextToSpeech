# TextToSpeech

## Requirements

The requirements.txt assumes that your GPU is [CUDA-enabled](https://developer.nvidia.com/cuda-gpus) and that you are on Windows.

If your GPU is not CUDA-enabled, remove `+cu117` from the requirements.txt file before running the command below.

If you are on Linux or macOS, remove the line with `soundfile` and run `pip install sox_io`.

To install requirements, run the following command:

`pip install -r requirements.txt`

## Usage

To run the program, run the following command:

`python main.py`
