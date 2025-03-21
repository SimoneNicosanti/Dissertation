FROM python:3.9-bookworm

RUN pip install grpcio
RUN pip install grpcio-tools

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install PuLP
RUN apt-get install glpk-utils libglpk-dev -y

RUN pip install readerwriterlock
RUN pip install networkx

RUN groupadd -g 1234 customgroup && \
    useradd -m -u 1234 -g customgroup customuser


# USER customuser

CMD ["/bin/bash"]