ARG ULTRALYTICS_VERSION=8.3.27
FROM ultralytics/ultralytics:${ULTRALYTICS_VERSION}


## Need it because of litert
RUN pip install "numpy==2.0"
RUN pip install mypy
RUN pip install psutil
RUN pip install pandas
RUN pip install matplotlib
RUN pip install tensorflow-datasets
RUN pip install ai-edge-litert
RUN pip install prettytable
RUN pip install imageio

## User Settings
RUN groupadd -g 1234 customgroup && \
    useradd -m -u 1234 -g customgroup customuser
USER customuser

## Shell Settings
ENV SHELL=/usr/bin/bash