# edge-optimized-ai-assistant

# Installation

## Install docker
https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository


# Development

### Prerequisites
* A working [Ubuntu 24.10 LTS](https://releases.ubuntu.com/oracular/ubuntu-24.10-desktop-amd64.iso) host
* [Intel GPU Driver Installation](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-6.html#driver-installation)

### Install uv to manage python packages
```sh
sudo apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh

echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
```

### Install SW package
```sh
sudo apt install ffmpeg -y
```

### Install python packages
```sh
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

uv pip install ffmpeg transformers fastapi uvicorn python-multipart
```


### Create requirements.txt file
```sh
uv pip list

uv pip freeze > requirements.txt

```