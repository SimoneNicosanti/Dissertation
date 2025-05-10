import json

import networkx as nx

from CommonPlan.Plan import Plan
from CommonPlan.SolvedModelGraph import ComponentId
from CommonProfile.NodeId import NodeId


def build_plan(json_plan: str, model_name: str, deployer_id: str) -> Plan:
    print("Building Plan")
    print(json_plan)
    solved_graph: nx.DiGraph = nx.node_link_graph(
        json.loads(json_plan, object_hook=decode_complex_info)
    )
    print("Decoded Plan")
    return Plan(solved_graph, deployer_id)


def decode_complex_info(dct):
    if "type" in dct and dct["type"] == "NodeId":
        return NodeId(dct["node_name"])
    if "type" in dct and dct["type"] == "ComponentId":
        node_id = NodeId(dct["net_node_id"])
        return ComponentId(node_id, dct["component_idx"])
    return dct
