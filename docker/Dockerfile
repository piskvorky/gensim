FROM ubuntu:16.04

MAINTAINER Daniel Baptista Dias <danielbpdias@gmail.com>

ENV GENSIM_REPOSITORY https://github.com/danielbdias/gensim/archive
ENV GENSIM_VERSION develop

# Installs python, pip and setup tools
RUN apt-get update \
    && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    unzip \
    wget

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
    annoy \
    cython \
    gensim \
    jupyter \
    matplotlib \
    nltk \
    pandas \
    pyemd \
    sklearn

# Create gensim directory
RUN mkdir /gensim

# Download gensim from Github
RUN mkdir /gensim/download \
    && cd /gensim/download \
    && wget --quiet $GENSIM_REPOSITORY/$GENSIM_VERSION.zip \
    && unzip $GENSIM_VERSION.zip \
    && mv ./gensim-$GENSIM_VERSION/* /gensim \
    && rm -rf /gensim/download

# Run check script
RUN python /gensim/docker/check_fast_version.py

# Add running permission to startup script
RUN chmod +x /gensim/docker/start_jupyter_notebook.sh

# Define the starting command for this container and expose its running port
CMD sh -c '/gensim/docker/start_jupyter_notebook.sh 9000'
EXPOSE 9000