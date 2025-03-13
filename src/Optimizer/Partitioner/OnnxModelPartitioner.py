import os

import onnx

from Optimizer.Graph.Graph import NodeId
from Optimizer.Partitioner.ModelPartitioner import ModelPartitioner
from Optimizer.Plan.Plan import Plan


class OnnxModelPartitioner(ModelPartitioner):

    def __init__(self, model_path: str, divided_model_dir: str):
        super().__init__(model_path)

        self.onnx_model: onnx.ModelProto = onnx.load_model(self.model_path)
        self.divided_model_dir = divided_model_dir

    def partition_model(
        self,
        model_plan: Plan,
        model_name: str,
        deployment_server: NodeId,
    ) -> dict:

        for component_id in model_plan.get_all_components():

            if model_plan.is_component_only_input(component_id):
                continue

            if model_plan.is_component_only_output(component_id):
                continue

            input_names = model_plan.get_input_names_per_component(component_id)
            output_names = model_plan.get_output_names_per_component(component_id)

            output_path = (
                os.path.join(self.divided_model_dir, model_name)
                + f"_depl_{deployment_server}_server_{component_id.net_node_id}_comp_{component_id.component_idx}.onnx"
            )

            onnx.utils.extract_model(
                self.model_path,
                output_path,
                input_names,
                output_names,
            )

        pass
