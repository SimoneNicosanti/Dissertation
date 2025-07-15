import networkx as nx

from Optimizer.PostProcessing import ConnectedComponents


class PostProcessor:
    def __init__(self):
        pass

    @staticmethod
    def post_process_solution_graph(model_solved_graph: nx.MultiDiGraph):
        ConnectedComponents.ConnectedComponentsFinder.find_connected_components(
            model_solved_graph
        )

        pass
