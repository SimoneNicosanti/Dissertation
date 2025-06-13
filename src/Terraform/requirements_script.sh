#!/bin/bash

sudo apt-get update

## Installing python
sudo apt-get install python3-pip -y
sudo apt-get install python3-venv -y

## Installing rsync
sudo apt-get install rsync -y
cd

# ## Installing python dependencies
# pip install grpcio
# pip install grpcio-tools
# pip install onnx
# pip install onnxruntime
# pip install "numpy<2"
# pip install onnxslim
# pip install onnx-tool
# pip install opencv-python-headless
# pip install PyYAML
# pip install PuLP
# pip install readerwriterlock
# pip install networkx

# ## Installing optimizer
# sudo apt-get install libgl1 libglib2.0-0 -y
# sudo apt-get install glpk-utils libglpk-dev -y

## Installing and setting up docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

echo "Terminated"
