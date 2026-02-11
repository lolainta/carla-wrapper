#!/bin/bash
pushd /opt/carla
echo "Starting CarlaUE4 server..."
./CarlaUE4.sh -RenderOffScreen -carla-port=${CARLA_PORT:-2000} &
popd
sleep 2
pushd /app
uv run carla_wrapper/server.py
popd
