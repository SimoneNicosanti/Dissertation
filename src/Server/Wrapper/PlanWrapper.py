import ast

from Inference.InputInfo import ComponentInfo, ModelInfo


class PlanWrapper:
    def __init__(self, model_plan: dict[str]):
        self.model_name = model_plan["model_name"]
        self.deployer_id = model_plan["deployer_id"]
        self.plan_dict: dict[str] = model_plan["plan"]

    def is_only_input_component(self, comp_info: ComponentInfo) -> bool:
        for key in self.plan_dict.keys():
            key_tuple = ast.literal_eval(key)
            if str(key_tuple[0]) == str(comp_info.model_info.server_id):
                if str(key_tuple[1]) == str(comp_info.component_idx):
                    return self.plan_dict[key]["is_only_input"]

        return False

    def is_only_output_component(self, comp_info: ComponentInfo) -> bool:
        for key in self.plan_dict.keys():
            key_tuple = ast.literal_eval(key)
            if str(key_tuple[0]) == str(comp_info.model_info.server_id):
                if str(key_tuple[1]) == str(comp_info.component_idx):
                    return self.plan_dict[key]["is_only_output"]

        return False

    def get_connections_for_component(
        self, comp_info: ComponentInfo
    ) -> list[ComponentInfo]:
        pass

    def get_assigned_components(self, server_id: str) -> list[ComponentInfo]:
        model_info = ModelInfo(self.model_name, self.deployer_id, server_id)

        assigned_components: list[ComponentInfo] = []
        for key in self.plan_dict.keys():
            key_tuple = ast.literal_eval(key)
            if str(key_tuple[0]) == str(server_id):
                assigned_components.append(ComponentInfo(model_info, str(key_tuple[1])))

        return assigned_components
