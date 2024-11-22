from tensorflow.core.framework import tensor_pb2 as _tensor_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LayerRequest(_message.Message):
    __slots__ = ("modelName", "layerName", "requestId", "tensor")
    MODELNAME_FIELD_NUMBER: _ClassVar[int]
    LAYERNAME_FIELD_NUMBER: _ClassVar[int]
    REQUESTID_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    modelName: str
    layerName: str
    requestId: int
    tensor: _tensor_pb2.TensorProto
    def __init__(self, modelName: _Optional[str] = ..., layerName: _Optional[str] = ..., requestId: _Optional[int] = ..., tensor: _Optional[_Union[_tensor_pb2.TensorProto, _Mapping]] = ...) -> None: ...

class LayerResponse(_message.Message):
    __slots__ = ("hasValue", "result")
    HASVALUE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    hasValue: bool
    result: _tensor_pb2.TensorProto
    def __init__(self, hasValue: bool = ..., result: _Optional[_Union[_tensor_pb2.TensorProto, _Mapping]] = ...) -> None: ...
