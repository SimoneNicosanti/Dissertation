import tempfile

import onnx

from CommonIds.ComponentId import ComponentId
from CommonPlan.Plan import Plan
from ModelDivider.Divide.ModelPartitioner import ModelPartitioner


class OnnxModelPartitioner(ModelPartitioner):

    def __init__(self):
        super().__init__()

    def partition_model(
        self, model_plan: Plan, onnx_model: onnx.ModelProto
    ) -> dict[ComponentId, onnx.ModelProto]:

        with tempfile.TemporaryDirectory() as temp_dir:
            model_temp_file = temp_dir + "/whole_model.onnx"
            onnx.save_model(onnx_model, model_temp_file)

            divided_components = {}

            for idx, component_id in enumerate(model_plan.get_all_components()):

                if model_plan.is_component_only_input(component_id):
                    continue

                if model_plan.is_component_only_output(component_id):
                    continue
                
                ## This makes division faster with a single component
                if len(model_plan.get_all_components()) == 3 :
                    ## No component has been created but the whole model
                    divided_components[component_id] = onnx_model
                    continue

                input_names = model_plan.get_input_names_per_component(component_id)
                output_names = model_plan.get_output_names_per_component(component_id)

                extracted_model_temp_file = temp_dir + f"/comp_{idx}.onnx"
                onnx.utils.extract_model(
                    model_temp_file,
                    extracted_model_temp_file,
                    input_names=input_names,
                    output_names=output_names,
                )

                extracted_model: onnx.ModelProto = onnx.load_model(
                    extracted_model_temp_file
                )

                divided_components[component_id] = extracted_model

        return divided_components
