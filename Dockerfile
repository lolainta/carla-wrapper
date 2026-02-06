FROM carlasim/carla:0.9.16
USER root
RUN apt update && apt install -y --no-install-recommends git
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . .
RUN uv add /workspace/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl
RUN uv add git+https://github.com/lolainta/carla-api.git
