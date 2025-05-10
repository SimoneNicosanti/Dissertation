import ast
import copy

import networkx as nx

from CommonProfile.NodeId import NodeId


## Encodes a model profile in a nx.DiGraph exportable as json
def encode_model_profile(model_profile: nx.DiGraph) -> nx.DiGraph:

    encoded_graph = copy.deepcopy(model_profile)

    ## Change nodes ids to strings
    label_mapping = {node_id: node_id.node_name for node_id in encoded_graph.nodes}
    encoded_graph = nx.relabel_nodes(encoded_graph, label_mapping)

    ## Change regression representation
    regressor_coeffs: dict[tuple, float] = encoded_graph.graph["regressor"]["coef"]
    encoded_coeffs = {}
    for coeff_key, coeff_value in regressor_coeffs.items():
        encoded_key = str(tuple([node_id.node_name for node_id in coeff_key]))
        encoded_coeffs[encoded_key] = coeff_value
    encoded_graph.graph["regressor"]["coef"] = encoded_coeffs

    print("Encoded Graph")
    return encoded_graph


## Decodes a model profile previously encoded in an nx.DiGraph
## With custom data structures
def decode_model_profile(model_profile: nx.DiGraph) -> nx.DiGraph:

    decoded_graph = copy.deepcopy(model_profile)

    ## Change nodes ids to NodeId
    label_mapping = {node_name: NodeId(node_name) for node_name in decoded_graph.nodes}
    decoded_graph = nx.relabel_nodes(decoded_graph, label_mapping)

    ## Change regression representation
    regressor_coeffs: dict[tuple, float] = decoded_graph.graph["regressor"]["coef"]
    decoded_coeffs = {}
    for coeff_key, coeff_value in regressor_coeffs.items():

        evaluated_tuple = ast.literal_eval(coeff_key)
        decoded_key = tuple([NodeId(node_name) for node_name in evaluated_tuple])

        decoded_coeffs[decoded_key] = coeff_value
    decoded_graph.graph["regressor"]["coef"] = decoded_coeffs

    return decoded_graph
