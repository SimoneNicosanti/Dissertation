class SolvedGraphInfo:
    SOLVED = "solved"
    VALUE = "value"
    MODEL_NAME = "model_name"
    LATENCY_VALUE = "latency_value"
    ENERGY_VALUE = "energy_value"
    DEVICE_ENERGY_VALUE = "device_energy_value"

    pass


class SolvedNodeInfo:
    NET_NODE_ID = "net_node_id"
    GENERATOR = "generator"
    RECEIVER = "receiver"
    COMPONENT = "component"
    QUANTIZED = "quantized"

    pass


class SolvedEdgeInfo:
    TENSOR_NAME_LIST = "tensor_name_list"

    pass
