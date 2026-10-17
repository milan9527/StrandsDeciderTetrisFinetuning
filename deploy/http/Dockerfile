FROM ghcr.io/astral-sh/uv:0.9.17 AS uv
FROM python:3.11-slim-bookworm AS api
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home decider
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --extra api --no-dev
COPY backend ./backend
COPY finetuning ./finetuning
COPY scripts ./scripts
COPY finetune.py ./
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8080
CMD ["python", "-m", "finetuning.http_api", "--config", "/config/service.json", "--host", "0.0.0.0"]

FROM api AS gpu
USER root
RUN test "$(uname -m)" = x86_64 || (echo "The GPU training image requires linux/amd64" >&2; exit 1)
RUN uv venv /opt/training --python /usr/local/bin/python \
    && uv pip install --python /opt/training/bin/python torch==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && uv pip install --python /opt/training/bin/python -r finetuning/requirements.txt \
    && uv pip install --python /opt/training/bin/python flash-linear-attention==0.5.2 \
    && git clone https://github.com/strands-labs/strands-decider.git /opt/strands-decider \
    && git -C /opt/strands-decider checkout 6039cf8b64c6074e75804000c07ac3d4cce795f7 \
    && uv cache clean
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 10001
CMD ["python", "-m", "finetuning.worker", "--config", "/config/service.json"]
