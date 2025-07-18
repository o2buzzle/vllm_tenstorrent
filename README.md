# vllm_tenstorrent: Tenstorrent Backend Plugin for vLLM

## Overview

`vllm_tenstorrent` is a backend plugin for [vLLM](https://github.com/vllm-project/vllm) that enables inference on Tenstorrent hardware (NPU).

- **Supports**: vLLM >= 0.6.0
- **Hardware**: Tenstorrent NPU (e.g., Wormhole/N300)
- **Python**: >= 3.12

## Installation

1. **Clone this repository** and install dependencies:

```bash
# Clone vllm_tenstorrent and enter the directory
$ git clone <this-repo-url>
$ cd vllm_tenstorrent

# Create and activate a Python 3.12+ virtual environment using TT's create_venv.sh
$ export PYTHON_ENV_DIR=$(pwd)/venv
$ pushd ~/tt-metal && ./create_venv.sh && popd
$ source venv/bin/activate
```

2. **Set environment variables** (see `.vscode/settings.json` for examples):
- `TT_METAL_HOME`: Path to TT-Metal installation
- `ARCH_NAME`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `VLLM_TARGET_PLATFORM`, etc.

3. **Install plugin**
```bash
# Install package
$ pip install -e .
```

## Usage

### Offline Inference
Run the offline inference example:

```bash
python examples/offline_inference_tt.py
```

### OpenAI-Compatible Server
Start the OpenAI-compatible API server:

```bash
python examples/server_example_tt.py --model <model-name>
```

- See `examples/prompts.json` for sample prompts.
- Adjust `--max_num_seqs` and other arguments as needed.

## Project Structure

```
vllm_tenstorrent/
├── pyproject.toml
├── vllm_tenstorrent/           
│   ├── __init__.py
│   ├── platform.py
│   ├── model_executor/
│   │   └── model_loader/
│   │       └── tt_model_loader.py
│   └── worker/
│       ├── __init__.py
│       ├── tt_model_executor.py
│       └── tt_worker.py
└── examples/
    ├── offline_inference_tt.py
    ├── server_example_tt.py
    └── prompts.json
```

## Configuration
- See `.vscode/settings.json` for necessary environment variables.

## Development
- The plugin is structured to follow vLLM's extension points (`platform_plugins`, model loader, worker, etc.).
- To add support for new models, update the model registry and loader logic in `examples/offline_inference_tt.py` and `vllm_tenstorrent/model_executor/model_loader/tt_model_loader.py`.
