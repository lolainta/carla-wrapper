FROM carlasim/carla:0.9.16
USER root
RUN apt update && apt install -y --no-install-recommends git
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD --chown=carla https://github.com/carla-simulator/scenario_runner.git /scenario_runner
ADD --chown=carla https://github.com/lolainta/carla-api.git /carla-api


USER carla
WORKDIR /app
COPY --chown=carla ./pyproject.toml /app/pyproject.toml
RUN uv add /workspace/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
RUN uv add -r /scenario_runner/requirements.txt
RUN uv add /carla-api/
ENV PYTHONPATH=/scenario_runner/:/workspace/PythonAPI/carla/
COPY . .

ENV PORT=50051
ENV CARLA_PORT=2000

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/entrypoint.sh" ]
