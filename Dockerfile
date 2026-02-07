FROM carlasim/carla:0.9.16
USER root
RUN apt update && apt install -y --no-install-recommends git
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD https://github.com/carla-simulator/scenario_runner.git /scenario_runner
ADD https://github.com/lolainta/carla-api.git /carla-api


WORKDIR /app
COPY . .
RUN uv add /workspace/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
RUN uv add -r /scenario_runner/requirements.txt
RUN uv add /carla-api/
ENV PYTHONPATH=/scenario_runner/:/workspace/PythonAPI/carla/
