import json

import grpc

from Common import ConfigReader
from CommonProfile.ModelProfile import ModelProfile
from proto_compiled.common_pb2 import ModelId
from proto_compiled.model_profile_pb2 import ProfileRequest, ProfileResponse
from proto_compiled.model_profile_pb2_grpc import ModelProfileStub


class ModelProfileBuilder:
    def __init__(self):
        model_profiler_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "MODEL_PROFILER_ADDR"
        )
        model_profiler_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "MODEL_PROFILER_PORT"
        )
        self.model_profiler_chann = grpc.insecure_channel(
            "{}:{}".format(model_profiler_addr, model_profiler_port)
        )
        pass

    def build_model_profiles(self, models_ids: list[ModelId]) -> list[ModelProfile]:

        model_profiles: list[ModelProfile] = []

        model_profiler: ModelProfileStub = ModelProfileStub(self.model_profiler_chann)
        for model_id in models_ids:
            model_profile_req: ProfileRequest = ProfileRequest(model_id=model_id)
            model_profile_res: ProfileResponse = model_profiler.profile_model(
                model_profile_req
            )

            model_profile = ModelProfile.decode(
                json.loads(model_profile_res.model_profile)
            )

            model_profiles.append(model_profile)

        return model_profiles

        pass
