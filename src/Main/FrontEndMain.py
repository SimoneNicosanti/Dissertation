import argparse
from concurrent import futures

import grpc

from Common import ConfigReader
from CommonServer.PlanWrapper import PlanWrapper
from FrontEnd.FrontEndServer import FrontEndServer
from proto_compiled.common_pb2 import OptimizedPlan
from proto_compiled.optimizer_pb2 import OptimizationRequest
from proto_compiled.optimizer_pb2_grpc import OptimizationStub
from proto_compiled.server_pb2_grpc import add_InferenceServicer_to_server


def main(args):
    print("Asking For Plan Optimization")

    optimizer_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
        "addresses", "OPTIMIZER_ADDR"
    )
    optimizer_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "OPTIMIZER_PORT"
    )
    optimizer_stub: OptimizationStub = OptimizationStub(
        grpc.insecure_channel("{}:{}".format(optimizer_addr, optimizer_port))
    )
    opt_req = OptimizationRequest(
        model_names=args.model_list,
        latency_weight=args.latency_weight,
        energy_weight=args.energy_weight,
        device_max_energy=args.device_max_energy,
        requests_number=args.requests_number,
        deployment_server=args.deployment_server,
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

    frontend_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "FRONTEND_PORT"
    )
    add_InferenceServicer_to_server(frontend_servicer, server)
    server.add_insecure_port(f"[::]:{frontend_port}")

    server.start()
    print(f"Frontend Server running on port {frontend_port}...")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esempio di lista come argomento")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        help="Lista di numeri",
        default=["yolo11x-seg"],
    )
    parser.add_argument(
        "--latency-weight", type=float, help="Peso per la latenza", default=1.0
    )
    parser.add_argument(
        "--energy-weight", type=float, help="Peso per l'energia", default=0.0
    )
    parser.add_argument(
        "--device-max-energy", type=float, help="Energia massima device", default=-1.0
    )
    parser.add_argument(
        "--requests-number",
        type=int,
        nargs="+",
        help="Numero di richieste per modello",
        default=[1],
    )
    parser.add_argument(
        "--deployment-server", type=str, help="Server di deployment", default="0"
    )

    args = parser.parse_args()
    main(args)
