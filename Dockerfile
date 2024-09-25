FROM pytorch/pytorch

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    tqdm \
    matplotlib \
    scikit-learn \
    pytorch-lightning \
    torchvision

COPY . /app

CMD ["bash"]
