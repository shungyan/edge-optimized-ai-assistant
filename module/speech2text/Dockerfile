FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.5.20 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install --no-install-recommends -y libze1 intel-level-zero-gpu intel-opencl-icd \
       clinfo libze-dev intel-ocloc ffmpeg
       
# WORKDIR /app
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     UV_HTTP_TIMEOUT=1800 uv sync --frozen --no-install-project --no-dev

# ADD . /app
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen --no-dev

# ENV PATH="/app/.venv/bin:$PATH"

# ARG CLI_ARGS=""
# RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# CMD uv run whisper.py $CLI_ARGS

WORKDIR /app
ADD . /app

RUN uv venv --python 3.12 && \
    uv pip sync requirements.txt --index-strategy unsafe-best-match \
    --index https://download.pytorch.org/whl/nightly/xpu

ENV PATH="/app/.venv/bin:$PATH"

ARG CLI_ARGS=""
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
CMD uv run whisper.py $CLI_ARGS