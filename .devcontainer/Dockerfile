# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gdb \
    cmake \
    make \
    ninja-build \
    pkg-config \
    # MPI dependencies for hyper-scalability
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    # Additional tools
    git \
    curl \
    wget \
    vim \
    nano \
    tree \
    htop \
    unzip \
    zip \
    ca-certificates \
    gnupg \
    lsb-release \
    # Python for scripting and tooling
    python3 \
    python3-pip \
    python3-dev \
    # Additional libraries that might be useful
    libssl-dev \
    libcurl4-openssl-dev \
    libjson-c-dev \
    libxml2-dev \
    libeigen3-dev \
    # Node.js for MCP servers (will be installed via features)
    && rm -rf /var/lib/apt/lists/*

# Install specific CMake version (latest stable)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    echo 'deb https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash vscode && \
    usermod -aG sudo vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up workspace directory
RUN mkdir -p /workspace && \
    chown vscode:vscode /workspace

# Switch to non-root user
USER vscode

# Set working directory
WORKDIR /workspace

# Set environment variables for MPI
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Install Python packages that might be useful
RUN pip3 install --user \
    conan \
    vcpkg \
    numpy \
    matplotlib \
    pytest

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Set up shell environment
RUN echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc && \
    echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias la="ls -la"' >> ~/.bashrc && \
    echo 'alias uvx="uv run --with"' >> ~/.bashrc

# Expose commonly used ports
EXPOSE 8080 3000 5000

# Set default command
CMD ["/bin/bash"]