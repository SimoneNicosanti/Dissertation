FROM python:3.10-bookworm

## TensorFlow Dependencies
RUN pip install ultralytics
RUN pip install ai-edge-litert
RUN pip install tensorflow
RUN pip install keras_cv
## For compatibility we need numpy 1.x
RUN pip install 'numpy<2'

## Other Python requirements
RUN pip install psutil
RUN pip install pandas
RUN pip install matplotlib
RUN pip install prettytable
RUN pip install imageio
RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install keras-hub

## Terraform
RUN apt-get update && apt-get install -y \
    curl gnupg software-properties-common && \
    curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" > /etc/apt/sources.list.d/hashicorp.list && \
    apt-get update && apt-get install -y terraform && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

## GLP Cli
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz 
RUN tar -xf google-cloud-cli-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh --usage-reporting false --path-update true
ENV PATH="/google-cloud-sdk/bin:$PATH"
## Remember to run gcloud --init --console-only


## Other Dependencies
RUN pip install onnx
RUN pip install onnxruntime
RUN pip install onnxslim
RUN pip install onnxruntime-extensions
RUN pip install onnx-tools
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install PuLP
RUN apt-get install glpk-utils libglpk-dev -y


## Shell Settings
ENV SHELL=/usr/bin/bash
ENV PATH="/ultralytics/ultralytics:$PATH"

## User Settings
RUN groupadd -g 1234 customgroup && \
    useradd -m -u 1234 -g customgroup customuser
USER customuser