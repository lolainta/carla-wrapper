#!/bin/bash
pushd /workspace/
su carla -c "./CarlaUE4.sh -RenderOffScreen &"
sleep 2
popd
pushd /app
uv run carla_wrapper/server.py
popd
