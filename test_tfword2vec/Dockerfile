FROM ubuntu:16.04

MAINTAINER Irina Saparina <irisaparina@gmail.com>

# Installs python, pip and setup tools (with fixed versions)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.1ubuntu2 \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev=2.6.1-0.1ubuntu2.3 \
        libpng12-dev=1.2.54-1ubuntu1 \
        libzmq3-dev=4.1.4-7 \
        pkg-config=0.29.1-0ubuntu1 \
        rsync=3.1.1-3ubuntu1 \
        software-properties-common=0.96.20.7 \
        unzip=6.0-20ubuntu1 \
        zip=3.0-11 \
        zlib1g-dev=1:1.2.8.dfsg-2ubuntu4.1 \
        openjdk-8-jdk=8u131-b11-2ubuntu1.16.04.3 \
        openjdk-8-jre-headless=8u131-b11-2ubuntu1.16.04.3 \
        #python=3.6.2 \
        #python3-pip=8.1.1-2ubuntu0.4 \
        python3-dev \
        libopenblas-dev=0.2.18-1ubuntu1 \
        wget=1.17.1-1ubuntu1.1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Upgrade pip
#RUN python3 -m pip install --upgrade pip

#Install numpy, scipy
RUN pip3 --no-cache-dir install \
        matplotlib==2.0.2 \
        numpy==1.13.1 \
        scipy==0.19.1 

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.5.4
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.

RUN git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout r1.3
WORKDIR /tensorflow

ENV CI_BUILD_PYTHON python3

RUN sed -i '\@https://github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz@d' tensorflow/workspace.bzl
RUN tensorflow/tools/ci_build/builds/configured CPU \
    bazel build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip3 && \
    pip --no-cache-dir install --upgrade /tmp/pip3/tensorflow-*.whl && \
    rm -rf /tmp/pip3 && \
    rm -rf /root/.cache
# Clean up pip wheel and Bazel cache when done.

# TensorBoard
EXPOSE 6006
WORKDIR /

#Download and unzip gensim tfw2v branch
RUN wget https://github.com/saparina/gensim/archive/distributed_tfw2v.zip 
RUN unzip distributed_tfw2v.zip

#Install gensim
RUN cd /gensim-distributed_tfw2v \
    && python3 setup.py install

#Prepare for testing
RUN mkdir test_tfw2v
COPY questions-words.txt word2vec.py word2vec_ops.cc word2vec_kernels.cc test_distributed_tf.py text8.zip /test_tfw2v/
WORKDIR ./test_tfw2v
RUN TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 && unzip text8.zip
