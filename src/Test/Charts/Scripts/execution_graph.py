# DEVICE_PLANS_PATH = "../../Results/DevicePlan/plan.json"
import json
import os
import re
import sys

import networkx as nx
from matplotlib import pyplot as plt

DEVICE_EDGE_PLANS_PATH = "../../Results/DeviceEdgePlan/plan.json"
DEVICE_EDGE_CLOUD_PLANS_PATH = "../../Results/DeviceEdgeCloudPlan/plan.json"

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from CommonPlan.Plan import Plan

pattern = re.compile(
    r"^(?P<model_name>[^_]+)"  # model name until first underscore
    r"_lw_(?P<latency_weight>-?\d+(?:\.\d+)?)"
    r"_ew_(?P<energy_weight>-?\d+(?:\.\d+)?)"
    r"_dme_(?P<device_max_energy>-?\d+(?:\.\d+)?)"
    r"_mn_(?P<max_noises>-?\d+(?:\.\d+)?)"
    r"_dc_(?P<device_cpus>-?\d+(?:\.\d+)?)"
    r"_ec_(?P<edge_cpus>-?\d+(?:\.\d+)?)"
    r"_cc_(?P<cloud_cpus>-?\d+(?:\.\d+)?)"
    r"_db_(?P<device_bandwidth>-?\d+(?:\.\d+)?)"
    r"_eb_(?P<edge_bandwidth>-?\d+(?:\.\d+)?)"
    r"_cb_(?P<cloud_bandwidth>-?\d+(?:\.\d+)?)$"
)


def draw_execution_graph(execution_plan: Plan):
    exec_graph: nx.DiGraph = nx.DiGraph()

    comp_id_map = {"0": "Dev", "1": "Edge", "2": "Cloud"}
    for comp_id in execution_plan.get_all_components():
        next_comp_conns = execution_plan.find_next_connections(comp_id).keys()

        curr_comp_key = (
            comp_id_map[comp_id.net_node_id.node_name],
            comp_id.component_idx,
        )

        exec_graph.add_node(
            curr_comp_key,
            is_only_input=execution_plan.is_component_only_input(comp_id),
            is_only_output=execution_plan.is_component_only_output(comp_id),
        )

        for next_comp in next_comp_conns:
            next_comp_key = (
                comp_id_map[next_comp.net_node_id.node_name],
                next_comp.component_idx,
            )
            exec_graph.add_edge(curr_comp_key, next_comp_key)

    try:
        from networkx.drawing.nx_pydot import graphviz_layout

        pos = graphviz_layout(exec_graph, prog="dot")  # 'dot' gives hierarchical layout
    except ImportError:
        # fallback to spring layout if pygraphviz is not installed
        pos = nx.spring_layout(exec_graph)

    comp_colors = []
    for comp_key in exec_graph.nodes:

        if exec_graph.nodes[comp_key].get("is_only_input", False) or exec_graph.nodes[
            comp_key
        ].get("is_only_output", False):
            comp_colors.append("lightcoral")

        else:
            comp_colors.append("lightblue")

    plt.figure(figsize=(6, 6))
    nx.draw(
        exec_graph,
        pos,
        with_labels=True,
        arrows=True,
        node_size=1000,
        node_color=comp_colors,
        font_size=8,
        font_weight="bold",
    )
    plt.title("Execution Plan Graph")
    plt.subplots_adjust(top=0.5)  # leave enough space for the title
    plt.savefig("../Images/Execution_Graphs/execution_graph.svg")

    pass


def main():
    device_edge_cloud_plans = json.loads(open(DEVICE_EDGE_CLOUD_PLANS_PATH).read())

    for key in device_edge_cloud_plans.keys():
        match = pattern.match(key)
        if match:
            result = match.groupdict()
            # convert numeric fields to int/float
            for k, v in result.items():
                if k == "model_name":
                    continue
                result[k] = float(v) if "." in v else int(v)

            max_noise = result["max_noises"]
            latency_weight = result["latency_weight"]
            device_cpus = result["device_cpus"]
            edge_cpus = result["edge_cpus"]
            cloud_cpus = result["cloud_cpus"]

            if (
                max_noise == 0.5
                and latency_weight == 0.0
                and device_cpus == 1.0
                and edge_cpus == 1.0
                and cloud_cpus == 0.0
            ):
                curr_plan = Plan.decode(device_edge_cloud_plans[key])
                draw_execution_graph(curr_plan)
                print("Has been drawn")

    pass


if __name__ == "__main__":
    main()
