from concurrent import futures

import grpc

from CommonServer import PlanWrapper
from CommonServer.InferenceInfo import ModelInfo
from FrontEnd.FrontEndServer import FrontEndServer
from proto_compiled.optimizer_pb2 import OptimizationRequest
from proto_compiled.optimizer_pb2_grpc import OptimizationStub
from proto_compiled.server_pb2_grpc import add_InferenceServicer_to_server

FRONTEND_PORT = 50090


def main():
    print("Asking For Plan Optimization")
    optimizer_stub: OptimizationStub = OptimizationStub(
        grpc.insecure_channel("optimizer:50060")
    )
    opt_req = OptimizationRequest(
        model_names=["yolo11n-seg"],
        latency_weight=1,
        energy_weight=0,
        device_max_energy=1,
        requests_number=[1],
        deployment_server="0",
    )
    optimizer_stub.serve_optimization(opt_req)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    frontend_servicer = FrontEndServer()

    add_InferenceServicer_to_server(frontend_servicer, server)
    server.add_insecure_port(f"[::]:{FRONTEND_PORT}")

    server.start()
    print(f"Frontend Server running on port {FRONTEND_PORT}...")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
