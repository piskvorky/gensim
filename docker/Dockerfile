FROM ubuntu:16.04

MAINTAINER Daniel Baptista Dias <danielbpdias@gmail.com>

ENV GENSIM_REPOSITORY https://github.com/danielbdias/gensim/archive
ENV GENSIM_VERSION develop

# Installs python, pip and setup tools (with fixed versions)
RUN apt-get update \
    && apt-get install -y \
    cmake=3.5.1-1ubuntu3 \
    g++=4:5.3.1-1ubuntu1 \
    git=1:2.7.4-0ubuntu1 \
    libboost-all-dev=1.58.0.1ubuntu1 \
    mercurial=3.7.3-1ubuntu1 \
    python3=3.5.1-3 \
    python3-pip=8.1.1-2ubuntu0.4 \
    python3-setuptools=20.7.0-1 \
    unzip=6.0-20ubuntu1 \
    wget=1.17.1-1ubuntu1.1

# Setup python language
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Make some symlinks
RUN cd /usr/bin \
    && ln -s easy_install3 easy_install \
    && ln -s pip3 pip

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install \
    annoy==1.8.3 \
    cython==0.25.2 \
    gensim==2.0.0 \
    jupyter==1.0.0 \
    matplotlib==2.0.0 \
    nltk==3.2.2 \
    pandas==0.19.2 \
    pyemd==0.4.3 \
    sklearn==0.0

# Download WordRank
RUN mkdir /gensim_dependencies \
    && cd /gensim_dependencies \
    && git clone https://bitbucket.org/shihaoji/wordrank

# Create gensim directory
RUN mkdir /gensim

# Download gensim from Github
RUN mkdir /gensim/download \
    && cd /gensim/download \
    && wget --quiet $GENSIM_REPOSITORY/$GENSIM_VERSION.zip \
    && unzip $GENSIM_VERSION.zip \
    && mv ./gensim-$GENSIM_VERSION/* /gensim \
    && rm -rf /gensim/download

# Copy custom WordRank file to Github and Install it
RUN cp /gensim/docker/wordrank_install.sh /gensim_dependencies/wordrank/install.sh \
    && cd /gensim_dependencies/wordrank \
    && sh ./install.sh

# # Run check script
# RUN python /gensim/docker/check_fast_version.py

# # Add running permission to startup script
# RUN chmod +x /gensim/docker/start_jupyter_notebook.sh

# # Define the starting command for this container and expose its running port
# CMD sh -c '/gensim/docker/start_jupyter_notebook.sh 9000'
# EXPOSE 9000