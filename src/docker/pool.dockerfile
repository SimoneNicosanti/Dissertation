FROM python:3.10-bookworm

RUN pip install grpcio
RUN pip install grpcio-tools

CMD ["/bin/bash"]