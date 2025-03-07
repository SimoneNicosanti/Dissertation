from dataclasses import dataclass

from Graph.Graph import EdgeId, Graph, GraphInfo, NodeId


class SolvedNodeInfo(GraphInfo):
    def __init__(self, net_node_id: NodeId):
        self.net_node_id: NodeId = net_node_id
        self.node_component: tuple[NodeId, int] = None

    def get_component(self) -> tuple[NodeId, int]:
        return self.node_component

    def set_component(self, component: tuple[NodeId, int]):
        self.node_component = component


class SolvedEdgeInfo(GraphInfo):
    def __init__(self, net_edge_id: EdgeId, mod_edge_names: list[str]):
        self.net_edge_id: NodeId = net_edge_id
        self.mod_edge_names: list[str] = mod_edge_names

    def get_tensor_names(self) -> list[str]:
        return self.mod_edge_names


@dataclass(frozen=True, repr=False)
class ComponentId:
    net_node_id: NodeId
    component_idx: int

    def __repr__(self):
        return f"({self.net_node_id}, {self.component_idx})"


class SolvedModelGraph(Graph):
    def __init__(self, graph_name: str, problem_solved: bool, solution_value: float):
        super().__init__(graph_name)

        self.problem_solved: bool = problem_solved
        self.solution_value: float = solution_value

    def is_solved(self) -> bool:
        return self.problem_solved

    def get_solution_value(self) -> float:
        return self.solution_value

    def get_used_net_nodes(self) -> set[NodeId]:
        net_nodes_set = set()
        for mod_node_id in self.get_nodes_id():
            net_node_id = self.get_node_info(mod_node_id).net_node_id
            net_nodes_set.add(net_node_id)

        return net_nodes_set

    def get_all_components(self) -> set[ComponentId]:
        components_set = set()
        for mod_node_id in self.get_nodes_id():
            node_info: SolvedNodeInfo = self.get_node_info(mod_node_id)
            components_set.add(node_info.get_component())

        return components_set

    def get_all_nodes_in_component(self, component_id: ComponentId) -> set[NodeId]:
        nodes_set = set()
        for mod_node_id in self.get_nodes_id():
            node_info: SolvedNodeInfo = self.get_node_info(mod_node_id)
            if node_info.get_component() == component_id:
                nodes_set.add(mod_node_id)

        return nodes_set
