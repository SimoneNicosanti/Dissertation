import csv

from Graph.ModelGraph import ModelEdgeInfo, ModelGraph, ModelNodeInfo


def print_profile_csv(model_graph: ModelGraph, file_path: str) -> None:

    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "NodeName",
                "FLOPs",
                "WeightMemory[MB]",
                "OutMemory[MB]",
                "NextNodeName",
                "EdgeDataSize[MB]",
            ]
        )
        for node_id in model_graph.get_nodes_id():
            node_info: ModelNodeInfo = model_graph.nodes[node_id]
            printable_row = []

            printable_row.append(node_id.node_name)
            printable_row.append(node_info.model_node_flops)
            printable_row.append(node_info.model_node_weights_size)
            printable_row.append(node_info.model_node_outputs_size)

            for edge_id in model_graph.get_nexts_from_node(node_id):
                edge_info: ModelEdgeInfo = model_graph.get_edge_info(edge_id)
                for tensor_name in edge_info.tensor_names:
                    printable_row.append(edge_id.second_node_id.node_name)
                    printable_row.append(edge_info.model_edge_data_size)

            writer.writerow(printable_row)
