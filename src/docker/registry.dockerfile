FROM python:3.9-bookworm

RUN pip install grpcio
RUN pip install grpcio-tools

RUN pip install readerwriterlock

RUN groupadd -g 1234 customgroup && \
    useradd -m -u 1234 -g customgroup customuser
# USER customuser

CMD ["/bin/bash"]