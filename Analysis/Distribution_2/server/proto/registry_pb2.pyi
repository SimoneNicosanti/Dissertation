from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class LayerInfo(_message.Message):
    __slots__ = ("modelName", "layerName")
    MODELNAME_FIELD_NUMBER: _ClassVar[int]
    LAYERNAME_FIELD_NUMBER: _ClassVar[int]
    modelName: str
    layerName: str
    def __init__(self, modelName: _Optional[str] = ..., layerName: _Optional[str] = ...) -> None: ...

class LayerPosition(_message.Message):
    __slots__ = ("layerInfo", "layerHost", "layerPort")
    LAYERINFO_FIELD_NUMBER: _ClassVar[int]
    LAYERHOST_FIELD_NUMBER: _ClassVar[int]
    LAYERPORT_FIELD_NUMBER: _ClassVar[int]
    layerInfo: LayerInfo
    layerHost: str
    layerPort: int
    def __init__(self, layerInfo: _Optional[_Union[LayerInfo, _Mapping]] = ..., layerHost: _Optional[str] = ..., layerPort: _Optional[int] = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ("modelName",)
    MODELNAME_FIELD_NUMBER: _ClassVar[int]
    modelName: str
    def __init__(self, modelName: _Optional[str] = ...) -> None: ...
