import grpc
from proto.optimizer_pb2 import OptimizationRequest
from proto.optimizer_pb2_grpc import OptimizationStub


def main():
    print("HELLO")
    optimizer_stub: OptimizationStub = OptimizationStub(
        grpc.insecure_channel("optimizer:50060")
    )
    opt_req = OptimizationRequest(
        model_names=["yolo11n-seg", "yolo11l-seg", "yolo11x-seg_quant"],
        latency_weight=1,
        energy_weight=0,
        device_max_energy=1,
        requests_number=[1, 1, 1],
        deployment_server="0",
    )
    optimizer_stub.serve_optimization(opt_req)
    pass


if __name__ == "__main__":
    main()
