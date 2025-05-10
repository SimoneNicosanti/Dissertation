import json
import os

import networkx as nx
import pandas as pd

from Common import ConfigReader
from CommonModel.PoolInterface import PoolInterface
from CommonProfile import ProfileCoder
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

            model_profile: nx.DiGraph = OnnxModelProfiler().profile_model(
                onnx_model, model_name, {}
            )

            dataframe: pd.DataFrame = QuantizationProfile().profile_quantization(
                onnx_model,
                model_profile,
                max_quantizable=10,
                calibration_dataset=dataset,
                train_set_size=750,
                test_set_size=50,
                calibration_size=100,
                noise_test_size=1,
            )

            regressor, train_score, test_score = QuantizationRegressor.build_regressor(
                dataframe, 10, 2, 3
            )

            QuantizationRegressor.embed_regressor_in_profile(
                model_profile, dataframe, regressor, train_score, test_score
            )

            self.save_profile(model_name, model_profile)

            ## TODO Add Quantization Evaluation in profiling

        model_profile = self.read_profile(model_name)

        return ProfileResponse(model_profile=model_profile)

    pass

    def save_profile(
        self,
        model_name,
        model_graph: nx.DiGraph,
    ):
        profiles_dir = ConfigReader.ConfigReader().read_str(
            "model_profiler_dirs", "PROFILES_DIR"
        )

        encoded_profile: nx.DiGraph = ProfileCoder.encode_model_profile(model_graph)
        model_profile_path = os.path.join(profiles_dir, model_name + ".json")

        with open(model_profile_path, "w") as json_file:
            json.dump(nx.node_link_data(encoded_profile), json_file)
        pass

    def read_profile(self, model_name: str) -> nx.DiGraph:
        profiles_dir = ConfigReader.ConfigReader().read_str(
            "model_profiler_dirs", "PROFILES_DIR"
        )

        model_profile_path = os.path.join(profiles_dir, model_name + ".json")
        model_profile = None
        if os.path.isfile(model_profile_path):
            with open(model_profile_path, "r") as json_file:
                model_profile = json_file.read()
        return model_profile
