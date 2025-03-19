FROM python:3.10-bookworm

RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install onnx
RUN pip install onnxruntime
RUN pip install "numpy<2"
RUN pip install onnxslim
RUN pip install onnx-tool

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install PuLP
RUN apt-get install glpk-utils libglpk-dev -y

RUN pip install readerwriterlock
RUN pip install networkx

RUN groupadd -g 1234 customgroup && \
    useradd -m -u 1234 -g customgroup customuser

RUN mkdir -p /optimizer_data && chown -R customuser:customgroup /optimizer_data
RUN mkdir -p /optimizer_data/models && chown -R customuser:customgroup /optimizer_data/models
RUN mkdir -p /optimizer_data/model_profiles && chown -R customuser:customgroup /optimizer_data/model_profiles
RUN mkdir -p /optimizer_data/divided_models && chown -R customuser:customgroup /optimizer_data/divided_models
RUN mkdir -p /optimizer_data/plans && chown -R customuser:customgroup /optimizer_data/plans


# USER customuser

CMD ["/bin/bash"]