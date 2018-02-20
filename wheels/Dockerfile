ARG BASE_IMAGE
FROM ${BASE_IMAGE}
LABEL maintainer "Joan Puigcerver <joapuipe@gmail.com>"
ENV DOCKER=1

# Essential stuff
RUN apt-get update --fix-missing && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
      libssl-dev \
      software-properties-common

# Install CMake3
RUN add-apt-repository -y ppa:george-edison55/cmake-3.x && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake

# Extra repositories
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 2.7
RUN apt-get install -y --no-install-recommends \
    libpython2.7 \
    libpython2.7-dev \
    libpython2.7-minimal \
    libpython2.7-stdlib \
    python2.7-minimal

# Install Python 3.5
RUN apt-get install -y --no-install-recommends \
      libpython3.5 \
      libpython3.5-dev \
      libpython3.5-minimal \
      libpython3.5-stdlib \
      python3.5-minimal

# Install Python 3.6
RUN apt-get install -y --no-install-recommends \
      libpython3.6 \
      libpython3.6-dev \
      libpython3.6-minimal \
      libpython3.6-stdlib \
      python3.6-minimal

# Install pip
RUN apt-get install -y --no-install-recommends \
    python-pip \
    python3-pip

# Upgrade pip
RUN pip install --upgrade pip && \
    pip3 install --upgrade pip

# Install virtualenv
RUN pip2 install virtualenv && \
    pip3 install virtualenv
