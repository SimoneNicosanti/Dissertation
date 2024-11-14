FROM python:3.10

RUN pip install "numpy==1.26.4"
RUN pip install requests
RUN pip install ai-edge-litert
RUN pip install prettytable

SHELL [ "/usr/bin/bash" ]