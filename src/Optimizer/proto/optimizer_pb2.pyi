import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OptimizationRequest(_message.Message):
    __slots__ = ("model_names", "latency_weight", "energy_weight", "device_max_energy", "requests_number", "deployment_server")
    MODEL_NAMES_FIELD_NUMBER: _ClassVar[int]
    LATENCY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    ENERGY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_MAX_ENERGY_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_NUMBER_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_SERVER_FIELD_NUMBER: _ClassVar[int]
    model_names: _containers.RepeatedScalarFieldContainer[str]
    latency_weight: float
    energy_weight: float
    device_max_energy: float
    requests_number: _containers.RepeatedScalarFieldContainer[int]
    deployment_server: str
    def __init__(self, model_names: _Optional[_Iterable[str]] = ..., latency_weight: _Optional[float] = ..., energy_weight: _Optional[float] = ..., device_max_energy: _Optional[float] = ..., requests_number: _Optional[_Iterable[int]] = ..., deployment_server: _Optional[str] = ...) -> None: ...
