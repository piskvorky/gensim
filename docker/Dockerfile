FROM ubuntu:16.04

MAINTAINER Parul Sethi <parul1sethi@gmail.com>

ENV GENSIM_REPOSITORY https://github.com/parulsethi/gensim.git
ENV GENSIM_VERSION add_wordrank_in_docker

# Installs python, pip and setup tools (with fixed versions)
RUN apt-get update \
    && apt-get install -y \
    ant=1.9.6-1ubuntu1 \
    cmake=3.5.1-1ubuntu3 \
    default-jdk=2:1.8-56ubuntu2 \
    g++=4:5.3.1-1ubuntu1 \
    git=1:2.7.4-0ubuntu1 \
    libboost-all-dev=1.58.0.1ubuntu1 \
    libgsl-dev=2.1+dfsg-2 \
    mercurial=3.7.3-1ubuntu1 \
    python3=3.5.1-3 \
    python3-pip=8.1.1-2ubuntu0.4 \
    python3-setuptools=20.7.0-1 \
    python=2.7.11-1 \
    python-pip=8.1.1-2ubuntu0.4 \
    python-setuptools=20.7.0-1 \
    unzip=6.0-20ubuntu1 \
    wget=1.17.1-1ubuntu1.1 \
    subversion=1.9.3-2ubuntu1 \
    locales=2.23-0ubuntu9 \
    libopenblas-dev=0.2.18-1ubuntu1 \
    libboost-program-options-dev=1.58.0.1ubuntu1 \
    zlib1g-dev=1:1.2.8.dfsg-2ubuntu4.1

# Setup python language
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Upgrade pip
RUN pip2 install --upgrade pip
RUN pip3 install --upgrade pip

# Install dependencies
RUN pip2 install \
    cython==0.25.2 \
    jupyter==1.0.0 \
    matplotlib==2.0.0 \
    nltk==3.2.2 \
    pandas==0.19.2 \
    spacy==1.8.1 \
    git+https://github.com/mila-udem/blocks.git@7beb788f1fcfc78d56c59a5edf9b4e8d98f8d7d9 \
    -r https://raw.githubusercontent.com/mila-udem/blocks/stable/requirements.txt

RUN pip3 install \
    cython==0.25.2 \
    jupyter==1.0.0 \
    matplotlib==2.0.0 \
    nltk==3.2.2 \
    pandas==0.19.2 \
    spacy==1.8.1 \
    git+https://github.com/mila-udem/blocks.git@7beb788f1fcfc78d56c59a5edf9b4e8d98f8d7d9 \
    -r https://raw.githubusercontent.com/mila-udem/blocks/stable/requirements.txt

# avoid using old numpy version installed by blocks requirements
RUN pip2 install -U numpy
RUN pip3 install -U numpy

# Download english model of Spacy
python2 -m spacy download en
python3 -m spacy download en

# Download gensim from Github
RUN git clone $GENSIM_REPOSITORY \
    && cd /gensim \
    && git checkout $GENSIM_VERSION \
    && pip2 install .[test] \
    && python2 setup.py install \
    && pip3 install .[test] \
    && python3 setup.py install

# Create gensim dependencies directory
RUN mkdir /gensim/gensim_dependencies

# Set ENV variables for wrappers
ENV WR_HOME /gensim/gensim_dependencies/wordrank
ENV FT_HOME /gensim/gensim_dependencies/fastText
ENV MALLET_HOME /gensim/gensim_dependencies/mallet
ENV DTM_PATH /gensim/gensim_dependencies/dtm/dtm/main
ENV VOWPAL_WABBIT_PATH /gensim/gensim_dependencies/vowpal_wabbit/vowpalwabbit/vw

# For fixed version downloads of gensim wrappers dependencies
ENV WORDRANK_VERSION 44f3f7786f76c79c083dfad9d64e20bacfb4a0b0
ENV FASTTEXT_VERSION f24a781021862f0e475a5fb9c55b7c1cec3b6e2e
ENV MORPHOLOGICALPRIORSFORWORDEMBEDDINGS_VERSION ec2e37a3bcb8bd7b56b75b043c47076bc5decf22
ENV DTM_VERSION 67139e6f526b2bc33aef56dc36176a1b8b210056
ENV MALLET_VERSION 2.0.8
ENV VOWPAL_WABBIT_VERSION 69ecc2847fa0c876c6e0557af409f386f0ced59a

# Install custom dependencies

# Install mpich (a wordrank dependency) and remove openmpi to avoid mpirun conflict
RUN apt-get purge -y openmpi-common openmpi-bin libopenmpi1.10
RUN apt-get install -y mpich

# Install wordrank
RUN cd /gensim/gensim_dependencies \
    && git clone https://bitbucket.org/shihaoji/wordrank \
    && cp /gensim/docker/wordrank_install.sh /gensim/gensim_dependencies/wordrank/install.sh \
    && cd /gensim/gensim_dependencies/wordrank \
    && git checkout $WORDRANK_VERSION \
    && sh ./install.sh

# Install fastText
RUN cd /gensim/gensim_dependencies \
    && git clone https://github.com/facebookresearch/fastText.git \
    && cd /gensim/gensim_dependencies/fastText \
    && git checkout $FASTTEXT_VERSION \
    && make

# Install MorphologicalPriorsForWordEmbeddings
RUN cd /gensim/gensim_dependencies \
    && git clone https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings.git \
    && cd /gensim/gensim_dependencies/MorphologicalPriorsForWordEmbeddings \
    && git checkout $MORPHOLOGICALPRIORSFORWORDEMBEDDINGS_VERSION

# Install DTM
RUN cd /gensim/gensim_dependencies \
    && git clone https://github.com/blei-lab/dtm.git \
    && cd /gensim/gensim_dependencies/dtm/dtm \
    && git checkout $DTM_VERSION \
    && make

# Install Mallet
RUN mkdir /gensim/gensim_dependencies/mallet \
    && mkdir /gensim/gensim_dependencies/download \
    && cd /gensim/gensim_dependencies/download \
    && wget --quiet http://mallet.cs.umass.edu/dist/mallet-$MALLET_VERSION.zip \
    && unzip mallet-$MALLET_VERSION.zip \
    && mv ./mallet-$MALLET_VERSION/* /gensim/gensim_dependencies/mallet \
    && rm -rf /gensim/gensim_dependencies/download \
    && cd /gensim/gensim_dependencies/mallet \
    && ant

# Install Vowpal wabbit
RUN cd /gensim/gensim_dependencies \
    && git clone https://github.com/JohnLangford/vowpal_wabbit.git \
    && cd /gensim/gensim_dependencies/vowpal_wabbit \
    && git checkout $VOWPAL_WABBIT_VERSION \
    && make \
    && make install

# Start gensim

# Run check script
RUN python2 /gensim/docker/check_fast_version.py
RUN python3 /gensim/docker/check_fast_version.py

# Add running permission to startup script
RUN chmod +x /gensim/docker/start_jupyter_notebook.sh

# Define the starting command for this container and expose its running port
CMD sh -c '/gensim/docker/start_jupyter_notebook.sh 9000'
EXPOSE 9000
