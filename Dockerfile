# CUDA 11.8 runtime + cuDNN8 on Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Use bash for consistency
SHELL ["/bin/bash", "-lc"]

# Basic OS deps (add GUI deps for full OpenCV)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget curl ca-certificates build-essential git \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 libx11-6 libxfixes3 libxtst6 \
    libgtk-3-0 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create env
RUN conda create -y -n srtquant python=3.10 && conda clean -afy
ENV CONDA_DEFAULT_ENV=srtquant
ENV PATH=$CONDA_DIR/envs/srtquant/bin:$PATH

RUN conda install -c conda-forge opencv -y

# (Optional) faster pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Working directory
WORKDIR /workspace

# Copy requirements first
COPY requirements.txt /workspace/requirements.txt

# Install dependencies: six first, then latest torch/cu118 wheels, then reqs
RUN pip install --no-cache-dir six \
 && pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu118 \
 && pip install --no-cache-dir -r requirements.txt

# Copy your project
COPY . /workspace

# Install your package
RUN python setup.py develop

# Default shell
CMD ["/bin/bash"]