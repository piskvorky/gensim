FROM ubuntu:16.04

MAINTAINER Daniel Baptista Dias <danielbpdias@gmail.com>

# Installs python, pip and setup tools
RUN apt-get update \
    && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools

# Setup python language
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Make some symlinks
RUN cd /usr/bin \
    && ln -s easy_install3 easy_install \
    && ln -s pydoc3 pydoc \
    && ln -s python3-config python-config \
    && ln -s pip3 pip \
    && ln -s python3 python

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip3 install \
    cython \
    sklearn \
    pandas \
    matplotlib \
    nltk \
    pyemd \
    jupyter \
    gensim