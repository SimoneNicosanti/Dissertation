from Graph.Graph import EdgeId, NodeId


class SolvedProblemInfo:
    def __init__(self, problem_solved: bool, solution_value: float):
        self.problem_solved: bool = problem_solved
        self.solution_value: float = solution_value

        ## net_node_id -> list[mod_node_id]
        self.node_assignments: dict[NodeId, list[NodeId]] = {}

        ## net_edge_id -> list[mod_edge_id]
        self.edge_assignments: dict[EdgeId, list[EdgeId]] = {}

    def is_solved(self) -> bool:
        return self.problem_solved

    def get_solution_value(self) -> float:
        return self.solution_value

    def get_used_network_nodes(self) -> list[NodeId]:
        return list(self.node_assignments.keys())

    def get_assigned_nodes_for_network_node(self, net_node_id: NodeId) -> list[NodeId]:
        return self.node_assignments.get(net_node_id)

    def get_input_edges_for_network_node(self, net_node_id: NodeId) -> list[EdgeId]:
        ret_list = []
        for net_edge_id, mod_edge_id_list in self.edge_assignments.items():
            if (
                net_edge_id.second_node_id == net_node_id
                and net_edge_id.first_node_id != net_node_id
            ):
                ret_list.extend(mod_edge_id_list)

        return ret_list

    def get_output_edges_for_network_node(self, net_node_id: NodeId) -> list[EdgeId]:
        ret_list = []
        for net_edge_id, mod_edge_id_list in self.edge_assignments.items():
            if (
                net_edge_id.first_node_id == net_node_id
                and net_edge_id.second_node_id != net_node_id
            ):
                ret_list.extend(mod_edge_id_list)

        return ret_list

    def get_self_edges_for_network_node(self, net_node_id: NodeId) -> list[EdgeId]:
        ret_list = []
        for net_edge_id, mod_edge_id_list in self.edge_assignments.items():
            if (
                net_edge_id.first_node_id == net_node_id
                and net_edge_id.second_node_id == net_node_id
            ):
                ret_list.extend(mod_edge_id_list)

        return ret_list

    def put_node_assignment(self, net_node_id: NodeId, mod_node_id: NodeId):
        self.node_assignments.setdefault(net_node_id, [])
        self.node_assignments.get(net_node_id).append(mod_node_id)
        pass

    def put_edge_assignment(self, net_edge_id: NodeId, mod_edge_id: NodeId):
        self.edge_assignments.setdefault(net_edge_id, [])
        self.edge_assignments.get(net_edge_id).append(mod_edge_id)
        pass
