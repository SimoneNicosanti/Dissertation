from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ("layers", "prevLayers", "nextLayers")
    class PrevLayersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: LayerList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[LayerList, _Mapping]] = ...) -> None: ...
    class NextLayersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: LayerList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[LayerList, _Mapping]] = ...) -> None: ...
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    PREVLAYERS_FIELD_NUMBER: _ClassVar[int]
    NEXTLAYERS_FIELD_NUMBER: _ClassVar[int]
    layers: LayerList
    prevLayers: _containers.MessageMap[str, LayerList]
    nextLayers: _containers.MessageMap[str, LayerList]
    def __init__(self, layers: _Optional[_Union[LayerList, _Mapping]] = ..., prevLayers: _Optional[_Mapping[str, LayerList]] = ..., nextLayers: _Optional[_Mapping[str, LayerList]] = ...) -> None: ...

class LayerList(_message.Message):
    __slots__ = ("layers",)
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    layers: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, layers: _Optional[_Iterable[str]] = ...) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ("hostName", "portNum")
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    PORTNUM_FIELD_NUMBER: _ClassVar[int]
    hostName: str
    portNum: int
    def __init__(self, hostName: _Optional[str] = ..., portNum: _Optional[int] = ...) -> None: ...

class LayerInfo(_message.Message):
    __slots__ = ("modelName", "layerName")
    MODELNAME_FIELD_NUMBER: _ClassVar[int]
    LAYERNAME_FIELD_NUMBER: _ClassVar[int]
    modelName: str
    layerName: str
    def __init__(self, modelName: _Optional[str] = ..., layerName: _Optional[str] = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ("modelName",)
    MODELNAME_FIELD_NUMBER: _ClassVar[int]
    modelName: str
    def __init__(self, modelName: _Optional[str] = ...) -> None: ...
