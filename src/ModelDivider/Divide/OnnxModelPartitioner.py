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

        divided_components = {}

        extractor: onnx.utils.Extractor = onnx.utils.Extractor(onnx_model)

        for component_id in model_plan.get_all_components():

            if model_plan.is_component_only_input(component_id):
                continue

            if model_plan.is_component_only_output(component_id):
                continue

            input_names = model_plan.get_input_names_per_component(component_id)
            output_names = model_plan.get_output_names_per_component(component_id)

            extracted_model: onnx.ModelProto = extractor.extract_model(
                input_names, output_names
            )

            divided_components[component_id] = extracted_model

        return divided_components
