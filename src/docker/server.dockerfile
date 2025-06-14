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
RUN pip install psutil

RUN groupadd -g 1234 customgroup && \
    useradd -m -u 1234 -g customgroup customuser

RUN mkdir -p /server_data && chown -R customuser:customgroup /server_data

## Iperf3 config
RUN apt-get update
RUN apt-get install iperf3 -y
RUN pip install iperf3

RUN apt-get install iproute2 -y

# USER customuser

# Inference port
EXPOSE 50004 
# Assignment port
EXPOSE 50005
# Ping port
EXPOSE 50006
# Frontend port
EXPOSE 50008


CMD ["/bin/bash"]