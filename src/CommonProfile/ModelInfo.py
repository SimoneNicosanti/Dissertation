class ModelNodeInfo:
    FLOPS = "flops"
    WEIGHTS_SIZE = "weights_size"
    OUTPUTS_SIZE = "outputs_size"
    IDX = "idx"
    GENERATOR = "generator"
    RECEIVER = "receiver"
    QUANTIZABLE = "quantizable"

    pass


class ModelEdgeInfo:
    TOT_TENSOR_SIZE = "tot_tensor_size"
    TENSOR_NAME_LIST = "tensor_name_list"

    pass

class ModelGraphInfo:
    NAME = "name"
    TENSOR_SIZE_DICT = "tensor_size_dict"
