FROM python:3.9-bookworm

RUN pip install grpcio
RUN pip install grpcio-tools

RUN pip install readerwriterlock

RUN apt update 
RUN apt install screen -y

RUN pip install networkx

RUN groupadd -g 1234 customgroup && \
    useradd -m -u 1234 -g customgroup customuser
# USER customuser

## Registry port
EXPOSE 50000
# StatePool port
EXPOSE 50005

CMD ["/bin/bash"]