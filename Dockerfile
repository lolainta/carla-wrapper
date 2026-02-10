FROM ubuntu:22.04 AS carla

RUN <<EOF
    apt update
    apt install -y --no-install-recommends wget ca-certificates
    rm -rf /var/lib/apt/lists/*
EOF

RUN <<EOF
    mkdir -p /opt/carla
    wget -qO /opt/carla.tar.gz https://tiny.carla.org/carla-0-9-16-linux
    tar -xzf /opt/carla.tar.gz -C /opt/carla
    rm /opt/carla.tar.gz
EOF

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -m -u 1000 carla

COPY --from=carla --chown=carla:carla /opt/carla /opt/carla
RUN <<EOF
    apt update
    apt install -y \
        git \
        ca-certificates \
        xserver-xorg
    rm -rf /var/lib/apt/lists/*
EOF




COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ADD https://github.com/carla-simulator/scenario_runner.git /opt/scenario_runner

USER carla
WORKDIR /app
COPY --chown=carla:carla ./pyproject.toml .
COPY --chown=carla:carla ./uv.lock .
RUN uv sync --locked
RUN uv add /opt/carla/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
RUN uv add -r /opt/scenario_runner/requirements.txt
ENV PYTHONPATH=/opt/scenario_runner/:/opt/carla/PythonAPI/carla/
COPY . .

ENV PORT=50051
ENV CARLA_PORT=2000

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/entrypoint.sh" ]
