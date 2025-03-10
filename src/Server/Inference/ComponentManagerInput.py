from Inference.ComponentManager import ComponentManager
from Inference.InferenceInfo import ComponentInfo


class ComponentManagerInput(ComponentManager):

    def __init__(self, component_info: ComponentInfo, input_names, output_names):
        super().__init__(component_info, input_names, output_names)
        pass
