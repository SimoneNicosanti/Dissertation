## Compile Server Services
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./inference.proto
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./assignee.proto

python -m grpc_tools.protoc -I./ --python_out=../Optimizer/proto --pyi_out=../Optimizer/proto --grpc_python_out=../Optimizer/proto ./optimizer.proto
python -m grpc_tools.protoc -I./ --python_out=../FrontEnd/proto --pyi_out=../FrontEnd/proto --grpc_python_out=../FrontEnd/proto ./optimizer.proto


python -m grpc_tools.protoc -I./ --python_out=../ModelPool/proto --pyi_out=../ModelPool/proto --grpc_python_out=../ModelPool/proto ./pool.proto
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./pool.proto

python -m grpc_tools.protoc -I./ --python_out=../Registry/proto --pyi_out=../Registry/proto --grpc_python_out=../Registry/proto ./register.proto
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./register.proto

## Compile common
python -m grpc_tools.protoc -I./ --python_out=../ModelPool/proto --pyi_out=../ModelPool/proto --grpc_python_out=../ModelPool/proto ./common.proto
python -m grpc_tools.protoc -I./ --python_out=../Registry/proto --pyi_out=../Registry/proto --grpc_python_out=../Registry/proto ./common.proto
python -m grpc_tools.protoc -I./ --python_out=../Server/proto --pyi_out=../Server/proto --grpc_python_out=../Server/proto ./common.proto
