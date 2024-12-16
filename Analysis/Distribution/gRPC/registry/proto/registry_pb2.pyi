from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ("mainModelName", "subModelIdx", "outputsNames")
    MAINMODELNAME_FIELD_NUMBER: _ClassVar[int]
    SUBMODELIDX_FIELD_NUMBER: _ClassVar[int]
    OUTPUTSNAMES_FIELD_NUMBER: _ClassVar[int]
    mainModelName: str
    subModelIdx: int
    outputsNames: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, mainModelName: _Optional[str] = ..., subModelIdx: _Optional[int] = ..., outputsNames: _Optional[_Iterable[str]] = ...) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ("hostName", "portNum")
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    PORTNUM_FIELD_NUMBER: _ClassVar[int]
    hostName: str
    portNum: int
    def __init__(self, hostName: _Optional[str] = ..., portNum: _Optional[int] = ...) -> None: ...

class LayerPosition(_message.Message):
    __slots__ = ("modelName", "layers", "serverInfo")
    MODELNAME_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    SERVERINFO_FIELD_NUMBER: _ClassVar[int]
    modelName: str
    layers: _containers.RepeatedScalarFieldContainer[str]
    serverInfo: ServerInfo
    def __init__(self, modelName: _Optional[str] = ..., layers: _Optional[_Iterable[str]] = ..., serverInfo: _Optional[_Union[ServerInfo, _Mapping]] = ...) -> None: ...

class LayerInfo(_message.Message):
    __slots__ = ("modelName", "layerName")
    MODELNAME_FIELD_NUMBER: _ClassVar[int]
    LAYERNAME_FIELD_NUMBER: _ClassVar[int]
    modelName: str
    layerName: str
    def __init__(self, modelName: _Optional[str] = ..., layerName: _Optional[str] = ...) -> None: ...
