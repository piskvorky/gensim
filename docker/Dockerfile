FROM ubuntu:16.04

MAINTAINER Ovidiu Dan <contact@ovidiudan.com>
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
RUN ln -s /usr/bin/easy_install3 /usr/bin/easy_install
RUN ln -s /usr/bin/pydoc3 /usr/bin/pydoc
RUN ln -s /usr/bin/python3-config /usr/bin/python-config
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip3 install cython
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install nltk
RUN pip3 install pyemd
RUN pip3 install jupyter
RUN pip3 install gensim

# RUN mkdir /storage/
# WORKDIR /storage/
# ADD https://raw.githubusercontent.com/jdwittenauer/ipython-notebooks/master/Gensim.ipynb /storage/
# VOLUME /storage/

# EXPOSE 9000

CMD ["python"]
# CMD ["ipython3", "notebook", "--no-browser", "--port", "9000", "--ip=*"]