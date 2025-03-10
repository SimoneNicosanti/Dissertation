import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceInfo(_message.Message):
    __slots__ = ("model_id", "input_files_paths")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_FILES_PATHS_FIELD_NUMBER: _ClassVar[int]
    model_id: _common_pb2.ModelId
    input_files_paths: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model_id: _Optional[_Union[_common_pb2.ModelId, _Mapping]] = ..., input_files_paths: _Optional[_Iterable[str]] = ...) -> None: ...

class InferenceReturn(_message.Message):
    __slots__ = ("model_id", "output_files_paths")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FILES_PATHS_FIELD_NUMBER: _ClassVar[int]
    model_id: _common_pb2.ModelId
    output_files_paths: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model_id: _Optional[_Union[_common_pb2.ModelId, _Mapping]] = ..., output_files_paths: _Optional[_Iterable[str]] = ...) -> None: ...

class CallbackInfo(_message.Message):
    __slots__ = ("model_id", "request_id", "output_files_paths")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FILES_PATHS_FIELD_NUMBER: _ClassVar[int]
    model_id: _common_pb2.ModelId
    request_id: _common_pb2.RequestId
    output_files_paths: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model_id: _Optional[_Union[_common_pb2.ModelId, _Mapping]] = ..., request_id: _Optional[_Union[_common_pb2.RequestId, _Mapping]] = ..., output_files_paths: _Optional[_Iterable[str]] = ...) -> None: ...
