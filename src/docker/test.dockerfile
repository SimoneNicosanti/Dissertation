FROM python:3.9-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    glpk-utils \
    libglpk-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    grpcio \
    grpcio-tools \
    readerwriterlock \
    networkx \
    "numpy<2" \
    onnx \
    onnxruntime \
    onnxslim \
    onnx-tool \
    pandas \
    scikit-learn \
    tqdm \
    opencv-python-headless \
    psutil \
    supervision



