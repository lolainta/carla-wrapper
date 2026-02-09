#!/bin/bash
pushd /workspace/
./CarlaUE4.sh -RenderOffScreen -carla-port=${CARLA_PORT:-2000} &
sleep 2
popd
pushd /app
uv run carla_wrapper/server.py
popd
