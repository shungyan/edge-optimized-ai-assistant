<a name="readme-top"></a>

# Edge Optimized AI Assistant

This project contains the software component and ingredients to enable optimized generative AI application on Intel Edge hardware

## Getting Started

### Prerequisites
* A working [Ubuntu 24.10 LTS](https://releases.ubuntu.com/oracular/ubuntu-24.10-desktop-amd64.iso) host
* [Intel GPU Driver Installation](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-6.html#driver-installation) (Optional)

### Installation

1\. Install Docker Compose from [(here)](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository). Add user to docker group, logging out and back in to apply change

```sh
sudo usermod -aG docker $USER
newgrp $USER
```

2\. Install FFmpeg

```sh
sudo apt install ffmpeg -y
```

3\. Clone repo & build docker image

*Note: If operating behind corporate firewall, setup the proxy settings, e.g. http_proxy, https_proxy, in Linux environment before continuing*

```sh
git clone https://github.com/huichuno/edge-optimized-ai-assistant.git && cd edge-optimized-ai-assistant/app/kiosk

docker compose -f docker-compose-lnl.yml build
```

4\. Start application and services

*Note: When launching the application for the first time, please allow a few minutes for the dependencies to download. You may check the containers log using lazydocker*

```sh
docker compose -f docker-compose-lnl.yml -f compose.override.yml up -d
```

5\. Launch browser and navigate to ***http://locahost:8080/*** on local machine or ***http://\<ip addr\>:8080/*** to access Open WebUI. Open WebUI is a self-hosted, open-source web interface that allows users to interact with LLMs locally

6\. Optional Installation
* [lazydocker](https://github.com/jesseduffield/lazydocker) - A simple terminal UI to visualize and interact with containers. After install, launch by executing the following command `~/.local/bin/lazydocker`

## How-To

### Stop application and services
```sh
docker compose -f docker-compose-lnl.yml -f compose.override.yml down
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Supported Platforms

| Codename | Product Name |
|--|--|
| Meteor Lake | Intel(R) Core(TM) Ultra Processors (Series 1) |
| Lunar Lake, Arrow Lake | Intel(R) Core(TM) Ultra Processors (Series 2) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## For Developer

### Install *uv* to manage Python packages
```sh
sudo apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh

echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
```

### Create an example Python project using *uv*
```sh
mkdir speech2text && cd speech2text
uv venv --python 3.12

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

uv pip install ffmpeg transformers fastapi uvicorn python-multipart
```

### Create a requirements.txt file
```sh
uv pip list

uv pip freeze > requirements.txt
```

### Sync virtual environment using requirements.txt file
```sh
uv pip sync requirements.txt \
  --index-strategy unsafe-best-match \
  --index https://download.pytorch.org/whl/xpu
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
