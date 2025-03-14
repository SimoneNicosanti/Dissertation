import json
from concurrent import futures

import grpc

from CommonServer.InferenceInfo import ModelInfo
from CommonServer.PlanWrapper import PlanWrapper
from FrontEnd.FrontEndServer import FrontEndServer
from proto_compiled.common_pb2 import OptimizedPlan
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
    optimized_plan: OptimizedPlan = optimizer_stub.serve_optimization(opt_req)
    plan_dict = optimized_plan.plans_map

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    frontend_servicer = FrontEndServer()

    for _, model_plan_str in plan_dict.items():
        plan_wrapper = PlanWrapper(model_plan_str)
        model_info = plan_wrapper.get_model_info()
        extreme_components = plan_wrapper.get_input_and_output_component()
        components_dict = {comp_info: None for comp_info in extreme_components}
        frontend_servicer.register_model(
            model_info, plan_wrapper, components_dict, None
        )

    add_InferenceServicer_to_server(frontend_servicer, server)
    server.add_insecure_port(f"[::]:{FRONTEND_PORT}")

    server.start()
    print(f"Frontend Server running on port {FRONTEND_PORT}...")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
