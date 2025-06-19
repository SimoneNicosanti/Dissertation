import argparse
import json
import time

import grpc

from CommonProfile.ModelProfile import ModelProfile
from proto_compiled.common_pb2 import ModelId
from proto_compiled.model_profile_pb2 import ProfileRequest, ProfileResponse
from proto_compiled.model_profile_pb2_grpc import ModelProfileStub


def main() :

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--model", type=str, help="Model Name", required=True)

    args = parser.parse_args()
    model_name = args.model

    model_profiler = ModelProfileStub(grpc.insecure_channel("localhost:50004"))

    profile_request = ProfileRequest(model_id=ModelId(model_name=model_name))

    start = time.perf_counter_ns()
    profile_response: ProfileResponse = model_profiler.profile_model(profile_request)
    end = time.perf_counter_ns()

    print("Profile Time >> ", (end - start) * 1e-9)




if __name__ == "__main__":
    main()