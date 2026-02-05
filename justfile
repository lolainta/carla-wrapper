proto:
    uv run -m grpc_tools.protoc -I=./proto --python_out=./carla_wrapper --grpc_python_out=./carla_wrapper ./proto/*.proto
