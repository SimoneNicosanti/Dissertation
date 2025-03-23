import os

import onnx

from Common import ConfigReader
from CommonProfile.NodeId import NodeId
from ModelManager.Divide.ModelPartitioner import ModelPartitioner
from CommonPlan.Plan import Plan


class OnnxModelPartitioner(ModelPartitioner):

    def __init__(self):
        super().__init__()

    def partition_model(
        self,
        model_plan: Plan,
        model_name: str,
        deployer_id: str,
    ) -> dict:

        model_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "model_manager_dirs", "MODELS_DIR"
        )
        model_path = os.path.join(model_dir, model_name) + ".onnx"

        divided_model_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "model_manager_dirs", "DIVIDED_MODELS_DIR"
        )

        for component_id in model_plan.get_all_components():

            if model_plan.is_component_only_input(component_id):
                continue

            if model_plan.is_component_only_output(component_id):
                continue

            input_names = model_plan.get_input_names_per_component(component_id)
            output_names = model_plan.get_output_names_per_component(component_id)

            output_path = (
                os.path.join(divided_model_dir, model_name)
                + f"_depl_{deployer_id}_server_{component_id.net_node_id}_comp_{component_id.component_idx}.onnx"
            )
            print(output_path)

            onnx.utils.extract_model(
                model_path,
                output_path,
                input_names,
                output_names,
            )

        pass
