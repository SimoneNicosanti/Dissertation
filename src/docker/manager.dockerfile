FROM python:3.9-bookworm

RUN pip install grpcio
RUN pip install grpcio-tools

RUN pip install readerwriterlock
RUN pip install networkx
RUN pip install "numpy<2"
RUN pip install onnx
RUN pip install onnxruntime
RUN pip install onnxslim
RUN pip install onnx-tool

CMD ["/bin/bash"]