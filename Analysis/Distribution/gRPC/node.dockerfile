FROM python:3.10-slim-bookworm

RUN pip install 'numpy<2'
RUN pip install psutil
RUN pip install pandas
RUN pip install matplotlib
RUN pip install tensorflow
RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install mypy-protobuf
RUN pip install ai-edge-litert

ENV SHELL=/usr/bin/bash