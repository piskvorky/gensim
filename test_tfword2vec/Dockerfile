FROM ubuntu:16.04

MAINTAINER Irina Saparina <irisaparina@gmail.com>

# Installs python, pip and setup tools (with fixed versions)
RUN apt-get update \
    && apt-get install -y \
    git=1:2.7.4-0ubuntu1 \
    python3=3.5.1-3 \
    python3-pip=8.1.1-2ubuntu0.4 \
    python=2.7.11-1 \
    python-pip=8.1.1-2ubuntu0.4 \
    libopenblas-dev=0.2.18-1ubuntu1 \
    unzip=6.0-20ubuntu1 \
    wget=1.17.1-1ubuntu1.1

# Upgrade pip
RUN python2 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade pip

#Install numpy, scipy and tensorflow (only CPU support)
RUN pip2 install --user numpy scipy tensorflow 
RUN pip3 install --user numpy scipy tensorflow 

#Download and unzip gensim tfw2v branch
RUN wget https://github.com/saparina/gensim/archive/distributed_tfw2v.zip
RUN unzip distributed_tfw2v.zip

#Install gensim
RUN cd /gensim-distributed_tfw2v \
    $$ pip2 install .[test] \
    && python2 setup.py install \
    && pip3 install .[test] \
    && python3 setup.py install

#Prepare for testing
RUN mkdir test_tfw2v
COPY test_distributed_tf.py text8.zip /test_tfw2v


