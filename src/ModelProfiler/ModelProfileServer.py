import json
import os

import networkx as nx
import pandas as pd

from Common import ConfigReader
from CommonModel.PoolInterface import PoolInterface
from CommonProfile.ModelProfile import ModelProfile, Regressor
from ModelProfiler.Profile.OnnxModelProfiler import OnnxModelProfiler
from ModelProfiler.Quantization import QuantizationRegressor
from ModelProfiler.Quantization.QuantizationProfile import QuantizationProfile
from proto_compiled.common_pb2 import ComponentId
from proto_compiled.model_profile_pb2 import ProfileRequest, ProfileResponse
from proto_compiled.model_profile_pb2_grpc import ModelProfileServicer


class ModelProfileServer(ModelProfileServicer):

    def __init__(self):
        self.model_pool_interface = PoolInterface()

        pass

    ## TODO Add profile request params
    def profile_model(self, profile_request: ProfileRequest, context):
        model_name = profile_request.model_id.model_name
        model_profile = self.read_profile(model_name)

        if model_profile is None:
            component_id = ComponentId(
                model_id=profile_request.model_id, server_id="", component_idx=""
            )

            onnx_model = self.model_pool_interface.retrieve_model(component_id)
            dataset = self.model_pool_interface.retrieve_calibration_dataset(
                profile_request.model_id
            )

            model_graph: nx.DiGraph = OnnxModelProfiler().profile_model(
                onnx_model, model_name, {}
            )
            print("Built Model Graph")

            train_set_size = 750
            test_set_size = 50
            calibration_size = 100
            noise_set_size = 1

            dataframe: pd.DataFrame = QuantizationProfile().profile_quantization(
                onnx_model,
                model_graph,
                max_quantizable=10,
                calibration_dataset=dataset,
                train_set_size=train_set_size,
                test_set_size=test_set_size,
                calibration_size=calibration_size,
                noise_test_size=noise_set_size,
            )
            print("Done Quantization Profile")

            regressor: Regressor = QuantizationRegressor.build_regressor(
                dataframe, train_set_size, test_set_size, 3
            )
            print("Built Regressor")

            model_profile = ModelProfile()
            model_profile.set_model_graph(model_graph)
            model_profile.set_regressor(regressor)

            self.save_profile(model_name, model_profile)

        model_profile_json = json.dumps(model_profile.encode())
        return ProfileResponse(model_profile=model_profile_json)

    pass

    def save_profile(
        self,
        model_name,
        model_profile: ModelProfile,
    ):
        profiles_dir = ConfigReader.ConfigReader().read_str(
            "model_profiler_dirs", "PROFILES_DIR"
        )

        encoded_profile = model_profile.encode()
        model_profile_path = os.path.join(profiles_dir, model_name + ".json")

        with open(model_profile_path, "w") as json_file:
            json.dump(encoded_profile, json_file)
        pass

    def read_profile(self, model_name: str) -> ModelProfile:
        profiles_dir = ConfigReader.ConfigReader().read_str(
            "model_profiler_dirs", "PROFILES_DIR"
        )

        model_profile_path = os.path.join(profiles_dir, model_name + ".json")
        model_profile = None
        if os.path.isfile(model_profile_path):
            with open(model_profile_path, "r") as json_file:
                encoded_profile: dict = json.load(json_file)
                model_profile = ModelProfile.decode(encoded_profile)
        return model_profile
