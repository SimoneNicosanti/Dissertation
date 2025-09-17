import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProducePlanRequest(_message.Message):
    __slots__ = ("models_ids", "latency_weight", "energy_weight", "device_max_energy", "requests_number", "max_noises", "start_server")
    MODELS_IDS_FIELD_NUMBER: _ClassVar[int]
    LATENCY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    ENERGY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_MAX_ENERGY_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_NUMBER_FIELD_NUMBER: _ClassVar[int]
    MAX_NOISES_FIELD_NUMBER: _ClassVar[int]
    START_SERVER_FIELD_NUMBER: _ClassVar[int]
    models_ids: _containers.RepeatedCompositeFieldContainer[_common_pb2.ModelId]
    latency_weight: float
    energy_weight: float
    device_max_energy: float
    requests_number: _containers.RepeatedScalarFieldContainer[int]
    max_noises: _containers.RepeatedScalarFieldContainer[float]
    start_server: str
    def __init__(self, models_ids: _Optional[_Iterable[_Union[_common_pb2.ModelId, _Mapping]]] = ..., latency_weight: _Optional[float] = ..., energy_weight: _Optional[float] = ..., device_max_energy: _Optional[float] = ..., requests_number: _Optional[_Iterable[int]] = ..., max_noises: _Optional[_Iterable[float]] = ..., start_server: _Optional[str] = ...) -> None: ...

class ProducePlanResponse(_message.Message):
    __slots__ = ("optimized_plan",)
    OPTIMIZED_PLAN_FIELD_NUMBER: _ClassVar[int]
    optimized_plan: str
    def __init__(self, optimized_plan: _Optional[str] = ...) -> None: ...

class DeploymentRequest(_message.Message):
    __slots__ = ("optimized_plan",)
    OPTIMIZED_PLAN_FIELD_NUMBER: _ClassVar[int]
    optimized_plan: str
    def __init__(self, optimized_plan: _Optional[str] = ...) -> None: ...

class DeploymentResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
