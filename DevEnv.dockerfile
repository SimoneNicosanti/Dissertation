ARG ULTRALYTICS_VERSION=8.3.27
FROM ultralytics/ultralytics:${ULTRALYTICS_VERSION}


RUN pip install psutil
RUN pip install pandas
RUN pip install matplotlib
RUN pip install tensorflow
RUN pip install ai-edge-litert
RUN pip install prettytable
RUN pip install imageio
RUN pip install rpyc
RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install mypy-protobuf


## Shell Settings
ENV SHELL=/usr/bin/bash
# ## Zsh Settings
# RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
#     -t jispwoso
## User Settings
RUN groupadd -g 1234 customgroup && \
    useradd -m -u 1234 -g customgroup customuser
USER customuser