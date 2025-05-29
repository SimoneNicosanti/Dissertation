import ast

from CommonPlan.SolvedModelGraph import ComponentId
from CommonProfile.NodeId import NodeId


class Plan:
    def __init__(self, model_name: str, plan_dict: dict[ComponentId, dict]):

        self.model_name = model_name
        self.plan_dict: dict[ComponentId, dict] = plan_dict

    def encode(self) -> dict:

        encoded_plan_dict = {}
        for component_id, model_plan in self.plan_dict.items():
            component_id_tuple = (
                component_id.model_name,
                component_id.net_node_id.node_name,
                component_id.component_idx,
            )
            encoded_component_id = str(component_id_tuple)
            encoded_plan_dict[encoded_component_id] = model_plan

        encoded_plan = {}

        encoded_plan["model_name"] = self.model_name
        encoded_plan["plan_dict"] = encoded_plan_dict

        return encoded_plan

    @staticmethod
    def decode(encoded_plan: dict) -> "Plan":

        model_name = encoded_plan["model_name"]
        plan_dict = {}
        for encoded_component_id in encoded_plan["plan_dict"].keys():
            component_id_tuple = ast.literal_eval(encoded_component_id)
            component_id = ComponentId(
                model_name=component_id_tuple[0],
                net_node_id=NodeId(node_name=component_id_tuple[1]),
                component_idx=component_id_tuple[2],
            )
            plan_dict[component_id] = encoded_plan["plan_dict"][encoded_component_id]

        return Plan(model_name, plan_dict)

    def is_component_only_input(self, key: ComponentId) -> bool:
        return self.plan_dict[key]["is_only_input"]

    def is_component_only_output(self, key: ComponentId) -> bool:
        return self.plan_dict[key]["is_only_output"]

    def get_input_names_per_component(self, key: ComponentId) -> list[str]:
        return self.plan_dict[key]["input_names"]

    def get_output_names_per_component(self, key: ComponentId) -> list[str]:
        return self.plan_dict[key]["output_connections"].keys()

    def get_all_components(self) -> list[ComponentId]:
        return list(self.plan_dict.keys())
