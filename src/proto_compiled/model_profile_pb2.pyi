import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProfileRequest(_message.Message):
    __slots__ = ("model_id", "profile_regression")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PROFILE_REGRESSION_FIELD_NUMBER: _ClassVar[int]
    model_id: _common_pb2.ModelId
    profile_regression: bool
    def __init__(self, model_id: _Optional[_Union[_common_pb2.ModelId, _Mapping]] = ..., profile_regression: bool = ...) -> None: ...

class ProfileResponse(_message.Message):
    __slots__ = ("model_profile",)
    MODEL_PROFILE_FIELD_NUMBER: _ClassVar[int]
    model_profile: str
    def __init__(self, model_profile: _Optional[str] = ...) -> None: ...
