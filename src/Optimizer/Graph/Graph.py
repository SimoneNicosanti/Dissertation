from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class NodeId:
    node_name: str

    def __repr__(self):
        return self.node_name


class ModelNodeInfo:
    FLOPS = "flops"
    WEIGHTS_SIZE = "weights_size"
    OUTPUTS_SIZE = "outputs_size"
    IDX = "idx"
    GENERATOR = "generator"
    RECEIVER = "receiver"

    pass


class ModelEdgeInfo:
    TOT_TENSOR_SIZE = "tot_tensor_size"
    TENSOR_NAME_LIST = "tensor_name_list"

    pass


class NetworkNodeInfo:
    FLOPS_PER_SEC = "flops_per_sec"
    COMP_ENERGY_PER_SEC = "comp_energy_per_sec"
    TRANS_ENERGY_PER_SEC = "trans_energy_per_sec"
    AVAILABLE_MEMORY = "available_memory"
    IDX = "idx"

    pass


class NetworkEdgeInfo:
    BANDWIDTH = "bandwidth"

    pass


class SolvedGraphInfo:
    SOLVED = "solved"
    VALUE = "value"

    pass


class SolvedNodeInfo:
    NET_NODE_ID = "net_node_id"
    GENERATOR = "generator"
    RECEIVER = "receiver"
    COMPONENT = "component"

    pass


class SolvedEdgeInfo:
    TENSOR_NAME_LIST = "tensor_name_list"

    pass
