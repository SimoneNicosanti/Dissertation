class NodeId:

    def __init__(self, node_id: str):
        self.node_id = node_id

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, value):
        return isinstance(value, NodeId) and self.node_id == value.node_id

    def get_node_id_str(self) -> str:
        return self.node_id


class EdgeId:
    def __init__(self, node_id_1: NodeId, node_id_2: NodeId):
        self.edge_id = (node_id_1, node_id_2)

    def __hash__(self):
        return hash(self.edge_id)

    def __eq__(self, value):
        return isinstance(value, EdgeId) and self.edge_id == value.edge_id

    def get_edge_id_str(self) -> str:
        return "({})>({})".format(
            self.edge_id[0].get_node_id_str(), self.edge_id[1].get_node_id_str()
        )

    def get_first_node_id(self) -> NodeId:
        return self.edge_id[0]

    def get_second_node_id(self) -> NodeId:
        return self.edge_id[1]
