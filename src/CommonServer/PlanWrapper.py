import ast

from CommonServer.InferenceInfo import ComponentInfo, ModelInfo


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
    ) -> dict[ComponentInfo, set[str]]:

        connections_dict: dict[ComponentInfo, set[str]] = {}
        for key in self.plan_dict.keys():
            if self.__is_same_key(key, comp_info):
                out_connections: dict[str, str] = self.plan_dict[key][
                    "output_connections"
                ]

                for tensor_name in out_connections.keys():
                    for comp_id in out_connections[tensor_name]:
                        next_comp_id_tuple = ast.literal_eval(comp_id)
                        next_comp_info = ComponentInfo(
                            comp_info.model_info,
                            str(next_comp_id_tuple[0]),
                            str(next_comp_id_tuple[1]),
                        )
                        connections_dict.setdefault(next_comp_info, set())
                        connections_dict[next_comp_info].add(tensor_name)

        return connections_dict

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(self.model_name, self.deployer_id)

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

    def get_input_for_component(self, component_info: ComponentInfo):
        for key in self.plan_dict.keys():
            if self.__is_same_key(key, component_info):
                print("Input Type >> ", type(self.plan_dict[key]["input_names"]))
                print("Elem Type >> ", type(self.plan_dict[key]["input_names"][0]))
                return self.plan_dict[key]["input_names"]

        return []

    def __is_same_key(self, key: tuple, comp_info: ComponentInfo) -> bool:
        key_tuple = ast.literal_eval(key)
        return str(key_tuple[0]) == str(comp_info.server_id) and str(
            key_tuple[1]
        ) == str(comp_info.component_idx)

    def get_input_and_output_component(self):
        model_info = ModelInfo(self.model_name, self.deployer_id)
        output_list = []
        for key in self.plan_dict.keys():
            key_tuple = ast.literal_eval(key)
            if (
                self.plan_dict[key]["is_only_output"]
                or self.plan_dict[key]["is_only_input"]
            ):
                output_list.append(
                    ComponentInfo(model_info, str(key_tuple[0]), str(key_tuple[1]))
                )
        return output_list
