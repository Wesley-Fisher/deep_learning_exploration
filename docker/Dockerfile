FROM tensorflow/tensorflow:latest-gpu

CMD mkdir /workspace
WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && \
    apt install -y \
      python3-pip \
      python3-tk \
      mesa-utils \
    && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install -r /workspace/requirements.txt


