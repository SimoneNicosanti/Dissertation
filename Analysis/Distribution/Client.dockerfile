FROM python

RUN pip install grpcio
RUN pip install requests
RUN pip install ai-edge-litert
RUN pip install prettytable

SHELL [ "/usr/bin/bash" ]