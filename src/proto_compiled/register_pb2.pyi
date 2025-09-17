import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ReachabilityInfo(_message.Message):
    __slots__ = ("ip_address", "assignment_port", "inference_port", "ping_port")
    IP_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENT_PORT_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_PORT_FIELD_NUMBER: _ClassVar[int]
    PING_PORT_FIELD_NUMBER: _ClassVar[int]
    ip_address: str
    assignment_port: int
    inference_port: int
    ping_port: int
    def __init__(self, ip_address: _Optional[str] = ..., assignment_port: _Optional[int] = ..., inference_port: _Optional[int] = ..., ping_port: _Optional[int] = ...) -> None: ...

class ServerId(_message.Message):
    __slots__ = ("server_id",)
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    server_id: str
    def __init__(self, server_id: _Optional[str] = ...) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ("reachability_info", "server_id")
    REACHABILITY_INFO_FIELD_NUMBER: _ClassVar[int]
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    reachability_info: ReachabilityInfo
    server_id: ServerId
    def __init__(self, reachability_info: _Optional[_Union[ReachabilityInfo, _Mapping]] = ..., server_id: _Optional[_Union[ServerId, _Mapping]] = ...) -> None: ...

class AllServerInfo(_message.Message):
    __slots__ = ("all_server_info",)
    ALL_SERVER_INFO_FIELD_NUMBER: _ClassVar[int]
    all_server_info: _containers.RepeatedCompositeFieldContainer[ServerInfo]
    def __init__(self, all_server_info: _Optional[_Iterable[_Union[ServerInfo, _Mapping]]] = ...) -> None: ...

class ServerState(_message.Message):
    __slots__ = ("server_id", "state")
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    server_id: str
    state: str
    def __init__(self, server_id: _Optional[str] = ..., state: _Optional[str] = ...) -> None: ...

class StateMap(_message.Message):
    __slots__ = ("network_profile",)
    NETWORK_PROFILE_FIELD_NUMBER: _ClassVar[int]
    network_profile: str
    def __init__(self, network_profile: _Optional[str] = ...) -> None: ...
