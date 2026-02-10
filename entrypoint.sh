#!/bin/bash
pushd /opt/carla
echo "Starting CarlaUE4 server..."
./CarlaUE4.sh -RenderOffScreen -nosound -carla-port=${CARLA_PORT:-2000} &
sleep 2
popd
pushd /app
uv run carla_wrapper/server.py
popd
