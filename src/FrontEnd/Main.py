
from proto.optimizer_pb2_grpc import OptimizationStub
from proto.optimizer_pb2 import OptimizationRequest, OptimizationResponse
import grpc




def main():
    print("HELLO")
    optimizer_stub : OptimizationStub = OptimizationStub(grpc.insecure_channel("optimizer:50060"))
    opt_req = OptimizationRequest(model_names=["yolo11n-seg"], latency_weight=1, energy_weight=0, device_max_energy=1, requests_number=[1], deployment_server="0")
    val : OptimizationResponse = optimizer_stub.serve_optimization(opt_req)
    pass


if __name__ == "__main__":
    main()