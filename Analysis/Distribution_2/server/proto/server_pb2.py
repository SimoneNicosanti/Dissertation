# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: server.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'server.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0cserver.proto\x12\x06server\x1a&tensorflow/core/framework/tensor.proto\"p\n\x0cLayerRequest\x12\x11\n\tmodelName\x18\x01 \x01(\t\x12\x11\n\tlayerName\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\x05\x12\'\n\x06tensor\x18\x04 \x01(\x0b\x32\x17.tensorflow.TensorProto\"J\n\rLayerResponse\x12\x10\n\x08hasValue\x18\x01 \x01(\x08\x12\'\n\x06result\x18\x02 \x01(\x0b\x32\x17.tensorflow.TensorProto2E\n\x06Server\x12;\n\nserveLayer\x12\x14.server.LayerRequest\x1a\x15.server.LayerResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'server_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_LAYERREQUEST']._serialized_start=64
  _globals['_LAYERREQUEST']._serialized_end=176
  _globals['_LAYERRESPONSE']._serialized_start=178
  _globals['_LAYERRESPONSE']._serialized_end=252
  _globals['_SERVER']._serialized_start=254
  _globals['_SERVER']._serialized_end=323
# @@protoc_insertion_point(module_scope)
