FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# Install Python e dipendenze base
RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip install grpcio
RUN pip install grpcio-tools

RUN pip install readerwriterlock
RUN pip install networkx
RUN pip install "numpy<2"
RUN pip install onnx
RUN pip install onnxslim
RUN pip install onnx-tool
RUN pip install pandas
RUN pip install scikit-learn

RUN pip install tqdm

RUN apt update 
RUN apt install screen -y

RUN pip install onnxruntime
RUN pip install onnxruntime-gpu


# Model pool port
EXPOSE 50002
# Model divider port
EXPOSE 50003
# Model profiler port
EXPOSE 50004
# Model Manager (NO LONGER VALID)
EXPOSE 50009

CMD ["/bin/bash"]