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
    vim \
    cmake \
    bats \
    ninja-build \
    libomp-dev \
    lld \
    python3-pip python3-numpy python3-wheel python3-setuptools python3-requests \
    libstdc++-14-dev \
    && apt-get clean

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 1 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 1


# Install Bazelisk on Linux AMD64
WORKDIR /workspace
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-amd64.deb
RUN apt install ./bazelisk-amd64.deb

WORKDIR /workspace
RUN git clone https://github.com/google/heir.git

# Create user
RUN adduser heiruser

# Entry point to get into the container
CMD ["/bin/bash"]
