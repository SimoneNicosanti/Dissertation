FROM python:3.10-bookworm

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
