FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /scrfd_train
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["sleep", "infinity"]