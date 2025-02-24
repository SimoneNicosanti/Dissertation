from Graph.Graph import Graph, GraphInfo, NodeId
from Graph.ModelGraph import ModelGraph


class AssignmentGraphInfo(GraphInfo):

    def __init__(
        self,
        info_dict: dict,
        server_id: NodeId,
        sub_graph: ModelGraph,
        sub_graph_idx: int,
    ):
        super().__init__(info_dict)

        self.server_id = server_id
        self.sub_graph = sub_graph
        self.sub_graph_idx = sub_graph_idx

    def get_sub_graph(self) -> ModelGraph:
        return self.sub_graph

    def get_server_id(self) -> NodeId:
        return self.server_id

    def get_block_idx(self) -> int:
        return self.sub_graph_idx


class AssignmentGraph(Graph):

    def __init__(
        self,
    ):
        super().__init__()

    pass
