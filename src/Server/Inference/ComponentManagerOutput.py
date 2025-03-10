from Inference.ComponentManager import ComponentManager


class ComponentManagerOutput(ComponentManager):

    def __init__(self, component_info, input_names, output_names):
        super().__init__(component_info, input_names, output_names)
