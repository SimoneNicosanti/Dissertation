FROM python:3.10-bookworm

RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install numpy
RUN pip install onnx
RUN pip install onnxruntime
RUN pip install opencv-python
RUN pip install grpcio
RUN pip install grpcio-tools

# RUN groupadd -g 1234 customgroup && \
#     useradd -m -u 1234 -g customgroup customuser
# USER customuser

CMD ["/bin/bash"]