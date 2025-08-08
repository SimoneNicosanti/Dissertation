import argparse
import time

import grpc

from Common import ConfigReader
from proto_compiled.common_pb2 import ModelId
from proto_compiled.server_pb2 import ExecutionProfileRequest, ExecutionProfileResponse
from proto_compiled.server_pb2_grpc import ExecutionProfileStub


def main():

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--model", type=str, help="Model Name", required=True)

    args = parser.parse_args()

    model_name = args.model

    execution_profile_request = ExecutionProfileRequest(
        model_id=ModelId(model_name=model_name)
    )

    port = ConfigReader.ConfigReader("../../../config/config.ini").read_int(
        "ports", "EXECUTION_PROFILER_PORT"
    )
    profiler_stub = ExecutionProfileStub(grpc.insecure_channel(f"localhost:{port}"))

    start = time.perf_counter_ns()
    profiler_response: ExecutionProfileResponse = profiler_stub.profile_execution(
        execution_profile_request
    )
    end = time.perf_counter_ns()


if __name__ == "__main__":
    main()
