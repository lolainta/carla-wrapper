import grpc
from concurrent import futures
import time

from carla_api import carla_pb2, carla_pb2_grpc
import carla


class CarlaService(carla_pb2_grpc.CarlaSimServicer):
    def __init__(self):
        print("Connecting to CARLA...")
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print("Connected to CARLA")

    def Ping(self, request, context):
        return carla_pb2.Pong(msg="CARLA alive")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    carla_pb2_grpc.add_CarlaSimServicer_to_server(CarlaService(), server)

    server.add_insecure_port("[::]:50051")
    server.start()

    print("gRPC server running on port 50051")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Shutting down")
        server.stop(0)


if __name__ == "__main__":
    serve()
