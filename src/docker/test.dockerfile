FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# Install Python e dipendenze base
RUN apt-get update && apt-get install -y python3 python3-pip git

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    glpk-utils \
    libglpk-dev \
    screen \
    && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install Python dependencies
RUN pip install --no-cache-dir \
    grpcio \
    grpcio-tools \
    networkx \
    numpy \
    onnx \
    onnxruntime \
    readerwriterlock \
    pandas \
    tqdm \
    opencv-python-headless \
    psutil \
    supervision

RUN pip install onnxruntime-openvino
