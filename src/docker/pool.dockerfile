FROM python:3.10-bookworm

RUN pip install grpcio
RUN pip install grpcio-tools

RUN pip install readerwriterlock

CMD ["/bin/bash"]