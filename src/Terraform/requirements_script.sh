#!/bin/bash

apt-get update
apt-get install python3-pip -y
apt-get install python3-venv -y
cd
python3 -m venv my_venv


pip install grpcio
pip install grpcio-tools
pip install onnx
pip install onnxruntime
pip install "numpy<2"
pip install onnxslim
pip install onnx-tool
pip install opencv-python-headless
pip install PyYAML
pip install PuLP
pip install readerwriterlock
pip install networkx

apt-get install -y libgl1 libglib2.0-0
apt-get install glpk-utils libglpk-dev -y




