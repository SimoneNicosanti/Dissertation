import onnx
from Graph.Graph import EdgeId, NodeId
from Graph.ModelGraph import ModelGraph
from Partitioner.ModelPartitioner import ModelPartitioner


class OnnxPartitioner(ModelPartitioner):

    def __init__(self, model_path: str):
        super().__init__(model_path)

        self.onnx_model: onnx.ModelProto = onnx.load(self.model_path)

    def partition_model(self, sub_graphs: list[ModelGraph]) -> list[str]:
        for idx, sub_graph in enumerate(sub_graphs):

            input_names = self.__find_input_names(sub_graph.get_input_edges())
            output_names = self.__find_output_names(sub_graph.get_output_edges())

            output_path = self.model_path.replace(".onnx", f"_sub_{idx}.onnx")
            onnx.utils.extract_model(
                self.model_path,
                output_path,
                input_names,
                output_names,
            )
        pass

    def __find_input_names(self, input_edges: list[EdgeId]) -> list[str]:
        input_names = set()

        ## TODO Check This Thing... I do not like it
        for first_node in self.onnx_model.graph.node:
            first_node_id = NodeId(first_node.name)
            first_node_outs = [out_name for out_name in first_node.output]
            for second_node in self.onnx_model.graph.node:
                second_node_id = NodeId(second_node.name)
                second_node_input = [inp_name for inp_name in second_node.input]

                for edge in input_edges:
                    if (
                        edge.get_first_node_id() == first_node_id
                        and edge.get_second_node_id() == second_node_id
                    ):
                        for inp_name in second_node_input:
                            if inp_name in first_node_outs:
                                input_names.add(inp_name)

                for graph_input in self.onnx_model.graph.input:
                    if graph_input.name in second_node_input:
                        input_names.add(graph_input.name)
        print("Input ", input_names)
        return input_names

    def __find_output_names(self, output_edges: list[EdgeId]) -> list[str]:
        output_names = []

        for first_node in self.onnx_model.graph.node:
            first_node_id = NodeId(first_node.name)
            first_node_outs = [out_name for out_name in first_node.output]
            for second_node in self.onnx_model.graph.node:
                second_node_id = NodeId(second_node.name)
                second_node_input = [inp_name for inp_name in second_node.input]

                for edge in output_edges:
                    if (
                        edge.get_first_node_id() == first_node_id
                        and edge.get_second_node_id() == second_node_id
                    ):
                        for out_name in first_node_outs:
                            if out_name in second_node_input:
                                output_names.append(out_name)

            for graph_output in self.onnx_model.graph.output:
                if graph_output.name in first_node_outs:
                    output_names.append(graph_output.name)
        print("Output ", output_names)

        return output_names
