## Compile Server Services
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./server.proto

python -m grpc_tools.protoc -I./ --python_out=../Optimizer/proto --pyi_out=../Optimizer/proto --grpc_python_out=../Optimizer/proto ./optimizer.proto
