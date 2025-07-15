import copy

from CommonIds.ComponentId import ComponentId
from CommonIds.NodeId import NodeId


class Plan:
    def __init__(
        self,
        model_name: str,
        plan_dict: dict[ComponentId, dict],
        quantized_nodes: list[NodeId],
        latency_cost: float = -1,
        energy_cost: float = -1,
        device_energy: float = -1,
    ):

        self.model_name = model_name
        self.plan_dict: dict[ComponentId, dict] = plan_dict
        self.quantized_nodes: list[NodeId] = quantized_nodes

        self.latency_cost = latency_cost
        self.energy_cost = energy_cost
        self.device_energy = device_energy

    def encode(self) -> dict:

        encoded_plan_dict = {}
        for component_id, component_info in self.plan_dict.items():

            encoded_plan_dict[component_id.encode()] = copy.deepcopy(component_info)

            encoded_out_connections = {}
            for tensor_name, next_comp_list in component_info[
                "output_connections"
            ].items():
                next_comp_list: list[ComponentId]
                encoded_out_connections[tensor_name] = [
                    next_comp_id.encode() for next_comp_id in next_comp_list
                ]
                pass

            encoded_plan_dict[component_id.encode()][
                "output_connections"
            ] = encoded_out_connections

        encoded_plan = {}

        encoded_plan["model_name"] = self.model_name
        encoded_plan["plan_dict"] = encoded_plan_dict
        encoded_plan["quantized_nodes"] = [
            comp_id.encode() for comp_id in self.quantized_nodes
        ]
        encoded_plan["latency_cost"] = self.latency_cost
        encoded_plan["energy_cost"] = self.energy_cost
        encoded_plan["device_energy"] = self.device_energy

        return encoded_plan

    @staticmethod
    def decode(encoded_plan: dict) -> "Plan":

        model_name = encoded_plan["model_name"]
        plan_dict = {}
        for encoded_component_id, encoded_comp_info in encoded_plan[
            "plan_dict"
        ].items():
            component_id = ComponentId.decode(encoded_component_id)

            decoded_out_connections = {}
            for tensor_name, encoded_next_comp_list in encoded_comp_info[
                "output_connections"
            ].items():
                decoded_out_connections[tensor_name] = [
                    ComponentId.decode(encoded_next_comp_id)
                    for encoded_next_comp_id in encoded_next_comp_list
                ]
                pass

            encoded_comp_info["output_connections"] = decoded_out_connections

            plan_dict[component_id] = encoded_plan["plan_dict"][encoded_component_id]

        quantized_nodes = [
            NodeId.decode(node_name) for node_name in encoded_plan["quantized_nodes"]
        ]

        return Plan(
            model_name,
            plan_dict,
            quantized_nodes,
            encoded_plan["latency_cost"],
            encoded_plan["energy_cost"],
            encoded_plan["device_energy"],
        )

    def get_latency_cost(self) -> float:
        return self.latency_cost

    def get_energy_cost(self) -> float:
        return self.energy_cost

    def get_device_energy(self) -> float:
        return self.device_energy

    def get_quantized_nodes(self) -> list[NodeId]:
        return self.quantized_nodes

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

    def get_server_components(self, server_id: str) -> list[ComponentId]:
        return [
            key
            for key in self.plan_dict.keys()
            if key.net_node_id.node_name == server_id
        ]

    def get_model_info(self) -> str:
        return self.model_name

    def get_input_names_for_component(self, component_id: ComponentId) -> list[str]:
        return self.plan_dict[component_id]["input_names"]

    def find_next_connections(
        self, comp_id: ComponentId
    ) -> dict[ComponentId, list[str]]:
        connections_dict: dict[ComponentId, set[str]] = {}

        component_info: dict = self.plan_dict[comp_id]

        out_connections: dict[str, ComponentId] = component_info["output_connections"]
        for tensor_name, next_comp_id_list in out_connections.items():
            next_comp_id_list: list[ComponentId]
            for next_comp_id in next_comp_id_list:

                connections_dict.setdefault(next_comp_id, set())
                connections_dict[next_comp_id].add(tensor_name)

        return connections_dict

    def get_input_and_output_component(self) -> list[ComponentId]:

        output_list = []
        for key in self.plan_dict.keys():
            if self.is_component_only_input(key) or self.is_component_only_output(key):
                output_list.append(key)
        return output_list
