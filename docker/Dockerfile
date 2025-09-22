# Base image with Python and build tools
FROM ubuntu:24.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Default LLVM toolchain version; can be overridden at build time
ARG CLANG_VERSION=19

# Install essential build tools and Python
RUN apt-get update && \
    apt-get install -y \
    apt-utils \
    build-essential \
    curl \
    git \
    zip \
    unzip \
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
    python3-numpy \
    python3-wheel \
    python3-setuptools \
    python3-requests \
    libstdc++-14-dev \
    ca-certificates \
    gnupg \
    lsb-release \
    && CODENAME=$(lsb_release -cs) && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/${CODENAME}/ llvm-toolchain-${CODENAME}-${CLANG_VERSION} main" > /etc/apt/sources.list.d/llvm.list && \
    apt-get update && \
    apt-get install -y \
    clang-${CLANG_VERSION} \
    lld-${CLANG_VERSION} \
    libomp-${CLANG_VERSION}-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${CLANG_VERSION} 200 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${CLANG_VERSION} 200 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-${CLANG_VERSION} 200


# Install Bazelisk on Linux AMD64
WORKDIR /tmp
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-amd64.deb && \
    apt install -y ./bazelisk-amd64.deb && \
    rm bazelisk-amd64.deb

# Create non-root user and prepare home for bind mounts
RUN useradd --create-home --shell /bin/bash heiruser && \
    mkdir -p /home/heiruser/heir && \
    chown -R heiruser:heiruser /home/heiruser

# Default working directory points at the mounted repo path
WORKDIR /home/heiruser/heir

# Entry point to get into the container
CMD ["/bin/bash"]
