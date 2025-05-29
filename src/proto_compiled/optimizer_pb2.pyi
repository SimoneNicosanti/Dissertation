from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OptimizationRequest(_message.Message):
    __slots__ = ("models_profiles", "network_profile", "execution_profile_pool", "latency_weight", "energy_weight", "device_max_energy", "requests_number", "max_noises", "start_server")
    MODELS_PROFILES_FIELD_NUMBER: _ClassVar[int]
    NETWORK_PROFILE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_PROFILE_POOL_FIELD_NUMBER: _ClassVar[int]
    LATENCY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    ENERGY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_MAX_ENERGY_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_NUMBER_FIELD_NUMBER: _ClassVar[int]
    MAX_NOISES_FIELD_NUMBER: _ClassVar[int]
    START_SERVER_FIELD_NUMBER: _ClassVar[int]
    models_profiles: _containers.RepeatedScalarFieldContainer[str]
    network_profile: str
    execution_profile_pool: str
    latency_weight: float
    energy_weight: float
    device_max_energy: float
    requests_number: _containers.RepeatedScalarFieldContainer[int]
    max_noises: _containers.RepeatedScalarFieldContainer[float]
    start_server: str
    def __init__(self, models_profiles: _Optional[_Iterable[str]] = ..., network_profile: _Optional[str] = ..., execution_profile_pool: _Optional[str] = ..., latency_weight: _Optional[float] = ..., energy_weight: _Optional[float] = ..., device_max_energy: _Optional[float] = ..., requests_number: _Optional[_Iterable[int]] = ..., max_noises: _Optional[_Iterable[float]] = ..., start_server: _Optional[str] = ...) -> None: ...

class OptimizationResponse(_message.Message):
    __slots__ = ("optimized_plan",)
    OPTIMIZED_PLAN_FIELD_NUMBER: _ClassVar[int]
    optimized_plan: str
    def __init__(self, optimized_plan: _Optional[str] = ...) -> None: ...
