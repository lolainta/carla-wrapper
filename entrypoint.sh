#!/bin/bash
pushd /workspace/
echo "Starting CarlaUE4.sh with -RenderOffScreen and -carla-port=${CARLA_PORT:-2000}"
./CarlaUE4.sh -RenderOffScreen -carla-port=${CARLA_PORT:-2000} &
sleep 2
popd
pushd /app
uv run carla_wrapper/server.py
popd
