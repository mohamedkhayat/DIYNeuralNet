FROM cupy/cupy:latest
USER root
RUN apt-get update && apt-get upgrade -y \ 
    && apt-get install -y --no-install-recommends openssh-client\
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
