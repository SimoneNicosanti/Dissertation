#!/bin/bash

rm ../Server/proto/*.py*
rm ../Optimizer/proto/*.py*
rm ../FrontEnd/proto/*.py*
rm ../ModelPool/proto/*.py*
rm ../Registry/proto/*.py*


## Compile for Server
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./server.proto
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./pool.proto
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./register.proto

## Compile for Optimizer
python -m grpc_tools.protoc -I./ --python_out=../Optimizer/proto --pyi_out=../Optimizer/proto --grpc_python_out=../Optimizer/proto ./optimizer.proto
python -m grpc_tools.protoc -I./ --python_out=../Optimizer/proto --pyi_out=../Optimizer/proto --grpc_python_out=../Optimizer/proto ./pool.proto
python -m grpc_tools.protoc -I./ --python_out=../Optimizer/proto --pyi_out=../Optimizer/proto --grpc_python_out=../Optimizer/proto ./server.proto

## Compile for FrontEnd
python -m grpc_tools.protoc -I./ --python_out=../FrontEnd/proto --pyi_out=../FrontEnd/proto --grpc_python_out=../FrontEnd/proto ./optimizer.proto


## Compile for ModelPool
python -m grpc_tools.protoc -I./ --python_out=../ModelPool/proto --pyi_out=../ModelPool/proto --grpc_python_out=../ModelPool/proto ./pool.proto

## Compile for Registry
python -m grpc_tools.protoc -I./ --python_out=../Registry/proto --pyi_out=../Registry/proto --grpc_python_out=../Registry/proto ./register.proto



## Compiling Common File
python -m grpc_tools.protoc -I./ --python_out=../ModelPool/proto --pyi_out=../ModelPool/proto --grpc_python_out=../ModelPool/proto ./common.proto
python -m grpc_tools.protoc -I./ --python_out=../Registry/proto --pyi_out=../Registry/proto --grpc_python_out=../Registry/proto ./common.proto
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./common.proto
python -m grpc_tools.protoc -I./ --python_out=../Optimizer/proto --pyi_out=../Optimizer/proto --grpc_python_out=../Optimizer/proto ./common.proto
python -m grpc_tools.protoc -I./ --python_out=../FrontEnd/proto --pyi_out=../FrontEnd/proto --grpc_python_out=../FrontEnd/proto ./common.proto
