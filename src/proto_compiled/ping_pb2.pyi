import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PingMessage(_message.Message):
    __slots__ = ("ping_bytes",)
    PING_BYTES_FIELD_NUMBER: _ClassVar[int]
    ping_bytes: bytes
    def __init__(self, ping_bytes: _Optional[bytes] = ...) -> None: ...
