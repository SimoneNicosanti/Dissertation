from tensorflow.core.framework import tensor_pb2 as _tensor_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class ModelInput(_message.Message):
    __slots__ = ("requestId", "modelName", "layerName", "tensor")
    REQUESTID_FIELD_NUMBER: _ClassVar[int]
    MODELNAME_FIELD_NUMBER: _ClassVar[int]
    LAYERNAME_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    requestId: int
    modelName: str
    layerName: str
    tensor: _tensor_pb2.TensorProto
    def __init__(
        self,
        requestId: _Optional[int] = ...,
        modelName: _Optional[str] = ...,
        layerName: _Optional[str] = ...,
        tensor: _Optional[_Union[_tensor_pb2.TensorProto, _Mapping]] = ...,
    ) -> None: ...

class ModelOutput(_message.Message):
    __slots__ = ("hasValue", "result")

    class ResultEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _tensor_pb2.TensorProto
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[_tensor_pb2.TensorProto, _Mapping]] = ...,
        ) -> None: ...

    HASVALUE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    hasValue: bool
    result: _containers.MessageMap[str, _tensor_pb2.TensorProto]
    def __init__(
        self,
        hasValue: bool = ...,
        result: _Optional[_Mapping[str, _tensor_pb2.TensorProto]] = ...,
    ) -> None: ...
