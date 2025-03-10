import ast

from Inference.InferenceInfo import ComponentInfo, ModelInfo


class PlanWrapper:
    def __init__(self, model_plan: dict[str]):
        self.model_name = model_plan["model_name"]
        self.deployer_id = model_plan["deployer_id"]
        self.plan_dict: dict[str] = model_plan["plan"]

    def is_only_input_component(self, comp_info: ComponentInfo) -> bool:
        for key in self.plan_dict.keys():
            if self.__is_same_key(key, comp_info):
                return self.plan_dict[key]["is_only_input"]

        return False

    def is_only_output_component(self, comp_info: ComponentInfo) -> bool:
        for key in self.plan_dict.keys():
            if self.__is_same_key(key, comp_info):
                return self.plan_dict[key]["is_only_output"]

        return False

    def find_next_connections(
        self, comp_info: ComponentInfo
    ) -> dict[str, list[ComponentInfo]]:

        connections_dict = {}
        for key in self.plan_dict.keys():
            if self.__is_same_key(key, comp_info):
                out_connections: dict[str, str] = self.plan_dict[key][
                    "output_connections"
                ]

                for tensor_name in out_connections.keys():
                    connections_dict.setdefault(tensor_name, [])
                    for comp_id in out_connections[tensor_name]:
                        next_comp_id_tuple = ast.literal_eval(comp_id)
                        next_comp_info = ComponentInfo(
                            comp_info.model_info,
                            str(next_comp_id_tuple[0]),
                            str(next_comp_id_tuple[1]),
                        )

                        connections_dict[tensor_name].append(next_comp_info)

        return connections_dict

    def get_assigned_components(self, server_id: str) -> list[ComponentInfo]:
        model_info = ModelInfo(self.model_name, self.deployer_id)

        assigned_components: list[ComponentInfo] = []
        for key in self.plan_dict.keys():
            key_tuple = ast.literal_eval(key)
            if str(key_tuple[0]) == str(server_id):
                assigned_components.append(
                    ComponentInfo(model_info, server_id, str(key_tuple[1]))
                )

        return assigned_components

    def __is_same_key(self, key: tuple, comp_info: ComponentInfo) -> bool:
        key_tuple = ast.literal_eval(key)
        return str(key_tuple[0]) == str(comp_info.server_id) and str(
            key_tuple[1]
        ) == str(comp_info.component_idx)
