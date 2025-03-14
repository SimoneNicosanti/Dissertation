#!/bin/bash

rm ../proto_compiled/*.py*

python -m grpc_tools.protoc -I./ --python_out=../proto_compiled --pyi_out=../proto_compiled --grpc_python_out=../proto_compiled ./common.proto
python -m grpc_tools.protoc -I./ --python_out=../proto_compiled --pyi_out=../proto_compiled --grpc_python_out=../proto_compiled ./optimizer.proto
python -m grpc_tools.protoc -I./ --python_out=../proto_compiled --pyi_out=../proto_compiled --grpc_python_out=../proto_compiled ./pool.proto
python -m grpc_tools.protoc -I./ --python_out=../proto_compiled --pyi_out=../proto_compiled --grpc_python_out=../proto_compiled ./register.proto
python -m grpc_tools.protoc -I./ --python_out=../proto_compiled --pyi_out=../proto_compiled --grpc_python_out=../proto_compiled ./server.proto
python -m grpc_tools.protoc -I./ --python_out=../proto_compiled --pyi_out=../proto_compiled --grpc_python_out=../proto_compiled ./state_pool.proto
python -m grpc_tools.protoc -I./ --python_out=../proto_compiled --pyi_out=../proto_compiled --grpc_python_out=../proto_compiled ./ping.proto
