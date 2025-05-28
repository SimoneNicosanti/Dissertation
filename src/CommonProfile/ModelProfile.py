import ast
import copy

import networkx as nx
from networkx.readwrite import json_graph

from CommonProfile.NodeId import NodeId


class Regressor:

    def __init__(
        self,
    ) -> None:
        self.interactions: dict[tuple[NodeId], float] = {}
        self.intercept = None
        self.degree = None
        self.train_score = None
        self.test_score = None

    def set_scores(self, train_score, test_score) -> None:
        self.train_score = train_score
        self.test_score = test_score

    def set_degree(self, degree) -> None:
        self.degree = degree

    def put_interaction(
        self, interaction_key: tuple[NodeId], interaction_value: float
    ) -> None:
        self.interactions[interaction_key] = interaction_value

    def set_intercept(self, intercept: float) -> None:
        self.intercept = intercept

    def encode(
        self,
    ) -> dict:
        transformed_obj = {}

        ## Encoding Interactions
        transformed_obj["interactions"] = {}
        for interaction_key in self.interactions:
            transformed_key = tuple([node_id.encode() for node_id in interaction_key])
            transformed_key = str(transformed_key)
            transformed_obj["interactions"][transformed_key] = self.interactions[
                interaction_key
            ]

        ## Encoding others
        transformed_obj["intercept"] = self.intercept
        transformed_obj["degree"] = self.degree
        transformed_obj["train_score"] = self.train_score
        transformed_obj["test_score"] = self.test_score

        return transformed_obj

    @staticmethod
    def decode(transformed_obj: dict) -> "Regressor":
        regressor = Regressor()

        ## Decoding Interactions
        for transformed_key, interaction_value in transformed_obj[
            "interactions"
        ].items():
            eval_key = ast.literal_eval(transformed_key)
            interaction_key = tuple([NodeId(node_name) for node_name in eval_key])
            regressor.put_interaction(interaction_key, interaction_value)

        regressor.set_intercept(transformed_obj["intercept"])
        regressor.set_degree(transformed_obj["degree"])
        regressor.set_scores(
            transformed_obj["train_score"], transformed_obj["test_score"]
        )

        return regressor


class ModelProfile:

    def __init__(self) -> None:
        self.model_graph: nx.DiGraph = None
        self.regressor: Regressor = None

    def set_model_graph(self, model_graph: nx.DiGraph) -> None:
        self.model_graph = model_graph

    def get_model_graph(self) -> nx.DiGraph:
        return self.model_graph

    def get_model_name(self) -> str:
        return self.model_graph.graph["name"]

    def set_regressor(self, regressor: Regressor) -> None:
        self.regressor = regressor

    def encode(self) -> dict:
        graph_copy = copy.deepcopy(self.model_graph)

        ## Change nodes ids to strings
        label_mapping = {node_id: node_id.node_name for node_id in graph_copy.nodes}
        encoded_graph = nx.relabel_nodes(graph_copy, label_mapping)

        encoded_graph = json_graph.node_link_data(encoded_graph)

        encoded_regressor = self.regressor.encode()

        encoded_profile = {
            "graph": encoded_graph,
            "regressor": encoded_regressor,
        }

        return encoded_profile

    @staticmethod
    def decode(transformed_dict: dict) -> "ModelProfile":

        model_profile = ModelProfile()

        model_graph = json_graph.node_link_graph(transformed_dict["graph"])
        model_graph = nx.relabel_nodes(
            model_graph,
            {node_name: NodeId(node_name) for node_name in model_graph.nodes},
            copy=True,
        )
        model_profile.set_model_graph(model_graph)
        regressor = Regressor.decode(transformed_dict["regressor"])
        model_profile.set_regressor(regressor)
        return model_profile
