from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ModelBlockId(_message.Message):
    __slots__ = ("model_name", "server_id", "block_idx")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCK_IDX_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    server_id: str
    block_idx: str
    def __init__(self, model_name: _Optional[str] = ..., server_id: _Optional[str] = ..., block_idx: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
