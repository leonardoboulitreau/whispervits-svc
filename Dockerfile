FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

# set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda3/bin:$PATH

# install linux packages
RUN apt update && apt install -y python3.8
RUN apt-get update && apt-get install -y curl libsndfile1 libopenblas-dev
RUN rm -rf /var/lib/apt/lists/*

# install PyTorch
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv python3-wheel
RUN pip install torch==1.9.0+cu111 torchaudio==0.9.0 tensorboard -f https://download.pytorch.org/whl/torch_stable.html

# install conda and pip packages
COPY ./requirements.txt /opt/gg-voice-conversion/environment/
WORKDIR /opt/gg-voice-conversion/environment
RUN pip install --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# set working directory
RUN apt update && apt-get update && apt-get install -y nano && apt-get install -y git && pip install notebook && apt install python-is-python3 && python -m pip install ipykernel && python -m ipykernel install --user
WORKDIR /workdir