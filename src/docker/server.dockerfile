FROM python:3.10-bookworm

RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install numpy
RUN pip install onnx
RUN pip install onnxruntime
RUN pip install opencv-python-headless
RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install PyYAML

RUN pip install readerwriterlock

RUN groupadd -g 1234 customgroup && \
    useradd -m -u 1234 -g customgroup customuser

RUN mkdir -p /server_data && chown -R customuser:customgroup /server_data

# USER customuser

CMD ["/bin/bash"]