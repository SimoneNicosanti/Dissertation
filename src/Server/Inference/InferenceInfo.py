from dataclasses import dataclass


@dataclass(frozen=True)
class SharedTensorInfo:
    tensor_name: str
    tensor_type: str
    tensor_shape: list
    shared_memory_name: str


@dataclass(frozen=True)
class ModelInfo:
    model_name: str
    deployer_id: str


@dataclass(frozen=True)
class ComponentInfo:
    model_info: ModelInfo
    server_id: str
    component_idx: str


@dataclass(frozen=True)
class RequestInfo:
    requester_id: str
    request_idx: int
