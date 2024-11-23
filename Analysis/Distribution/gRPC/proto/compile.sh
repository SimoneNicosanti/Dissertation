## Compile Registry Services
python -m grpc_tools.protoc -I./ --python_out=../registry/proto --pyi_out=../registry/proto --grpc_python_out=../registry/proto ./registry.proto
python -m grpc_tools.protoc -I./ --python_out=../client/proto --pyi_out=../client/proto --grpc_python_out=../client/proto ./registry.proto
python -m grpc_tools.protoc -I./ --python_out=../server/proto --pyi_out=../server/proto --grpc_python_out=../server/proto ./registry.proto


## Compile Server Services
python -m grpc_tools.protoc -I. -I=$(python -c "import tensorflow as tf; import os; print(os.path.join(tf.sysconfig.get_include()))") --python_out=../server/proto --pyi_out=../server/proto --grpc_python_out=../server/proto ./server.proto
python -m grpc_tools.protoc -I. -I=$(python -c "import tensorflow as tf; import os; print(os.path.join(tf.sysconfig.get_include()))") --python_out=../client/proto --pyi_out=../client/proto --grpc_python_out=../client/proto ./server.proto