FROM ubuntu:16.04

MAINTAINER Daniel Baptista Dias <danielbpdias@gmail.com>

ENV GENSIM_REPOSITORY https://github.com/RaRe-Technologies/gensim/archive
ENV GENSIM_VERSION master

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
    unzip=6.0-20ubuntu1 \
    wget=1.17.1-1ubuntu1.1
    libboost-program-options-dev \
    libboost-python-dev \
    zlib1g-dev

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
    annoy \
    cython \
    jupyter \
    matplotlib \
    nltk \
    pandas \
    pyemd \
    sklearn \
    fasttext \
    testfixtures \
    unittest2 \
    git+git://github.com/mila-udem/fuel.git@stable \
    git+git://github.com/mila-udem/blocks.git@stable \
    -r https://raw.githubusercontent.com/mila-udem/blocks/stable/requirements.txt

# Create gensim directory and dependencies directory
RUN mkdir /gensim \
    && mkdir /gensim_dependencies

# Download gensim from Github
RUN mkdir /gensim/download \
    && cd /gensim/download \
    && wget --quiet $GENSIM_REPOSITORY/$GENSIM_VERSION.zip \
    && unzip $GENSIM_VERSION.zip \
    && mv ./gensim-$GENSIM_VERSION/* /gensim \
    && rm -rf /gensim/download \
    && cd /gensim \
    && python setup.py install

# Set ENV variables for wrappers
ENV FT_HOME gensim_dependencies/fastText
ENV WR_HOME gensim_dependencies/wordrank
ENV MALLET_HOME gensim_dependencies/mallet
ENV DTM_PATH gensim_dependencies/dtm/bin/dtm-linux64
ENV VOWPAL_WABBIT_PATH gensim_dependencies/vowpal_wabbit/vowpalwabbit/vw

# Install custom dependencies

# Install WordRank
RUN cd /gensim_dependencies \
    && git clone https://bitbucket.org/shihaoji/wordrank \
    && cp /gensim/docker/wordrank_install.sh /gensim_dependencies/wordrank/install.sh \
    && cd /gensim_dependencies/wordrank \
    && sh ./install.sh

# Install fastText
RUN cd /gensim_dependencies \
    && git clone https://github.com/facebookresearch/fastText.git \
    && cd /gensim_dependencies/fastText \
    && make

# Install MorphologicalPriorsForWordEmbeddings
RUN cd /gensim_dependencies \
    && git clone https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings.git

# Install DTM
RUN cd /gensim_dependencies \
    && git clone https://github.com/magsilva/dtm.git

# Install Mallet
RUN mkdir /gensim_dependencies/mallet \
    && mkdir /gensim_dependencies/download \
    && cd /gensim_dependencies/download \
    && wget --quiet http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip \
    && unzip mallet-2.0.8.zip \
    && mv ./mallet-2.0.8/* /gensim_dependencies/mallet \
    && rm -rf /gensim_dependencies/download \
    && cd /gensim_dependencies/mallet \
    && ant

# Install Vowpal wabbit
RUN cd /gensim_dependencies \
    && git clone https://github.com/JohnLangford/vowpal_wabbit.git
    && cd /gensim_dependencies/vowpal_wabbit \
    && make
    && make install

# Start gensim

# Run check script
RUN python /gensim/docker/check_fast_version.py

# Add running permission to startup script
RUN chmod +x /gensim/docker/start_jupyter_notebook.sh

# Define the starting command for this container and expose its running port
CMD sh -c '/gensim/docker/start_jupyter_notebook.sh 9000'
EXPOSE 9000