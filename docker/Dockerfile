# Start FROM pytorch image
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# Install linux packages
# gnupg required for Edge TPU install
# libsm6 required by libqxcb to create QT-based windows for visualization; set 'QT_DEBUG_PLUGINS=1' to test in docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl tar

# Create working directory
WORKDIR /vidnn

# Install pip packages
RUN pip install lightning
