# Base image with Python and build tools
FROM ubuntu:24.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and Python
RUN apt-get update && \
    apt-get install -y \
    apt-utils \
    build-essential \
    curl \
    git \
    zip \
    unzip \
    clang-19 \
    python3-dev \
    python3-pip \
    python3-venv \
    openjdk-17-jdk \
    wget \
    software-properties-common \
    pkg-config \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libcurl4-openssl-dev \
    libfreetype6-dev \
    libzmq3-dev \
    libtool \
    && apt-get clean
RUN apt-get install -y bats
# Use Python 3.10 explicitly
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN apt install -y vim emacs cmake

WORKDIR /workspace
RUN wget https://github.com/ninja-build/ninja/archive/refs/tags/v1.12.1.tar.gz && tar -xzvf v1.12.1.tar.gz && cd ninja-1.12.1 && cmake -S . -B build && cmake --build build --target install -j

# Install Bazelisk on Linux AMD64
WORKDIR /workspace
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-amd64.deb
RUN apt install ./bazelisk-amd64.deb

WORKDIR /workspace
RUN git clone https://github.com/google/heir.git

# Show most recent file changes; work on HEAD
WORKDIR /workspace/heir
RUN git diff --name-only HEAD~1

# Install required Python packages
RUN apt-get install -y python3-pip python3-numpy python3-wheel python3-setuptools python3-requests
#keras_preprocessing requests

# Create user
RUN adduser ec2user

# Symlink clang-19 -> clang, clang++-19 -> clang++
WORKDIR /usr/bin
RUN ln -s clang++-19 clang++
WORKDIR /usr/bin
RUN ln -s clang-19 clang

RUN apt-get install -y libstdc++-14-dev

# Entry point to get into the container
CMD ["/bin/bash"]
