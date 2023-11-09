FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 \
    python3-pip \
    python2.7 \
    autoconf \
    automake \
    cmake \
    curl \
    g++ \
    git \
    graphviz \
    libatlas3-base \
    libtool \
    make \
    pkg-config \
    sox \
    subversion \
    unzip \
    wget \
    zlib1g-dev

RUN ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install setuptools
RUN pip install --upgrade setuptools

RUN pip install --upgrade pip \ 
    numpy \ 
    pyparsing \ 
    jupyter \ 
    ninja

RUN git clone https://github.com/pykaldi/pykaldi.git

RUN apt-get install gfortran -y

RUN cd /pykaldi/tools \
   && ./check_dependencies.sh \
   &&  ./install_protobuf.sh \
   &&  ./install_clif.sh 

RUN git clone -b pykaldi_02 https://github.com/pykaldi/kaldi.git /pykaldi/tools/kaldi \
    && cd /pykaldi/tools/kaldi/tools \
    && ./extras/install_mkl.sh

RUN cd /pykaldi/tools \
    &&  ./install_kaldi.sh 

RUN cd /pykaldi \
    && KALDI_DIR=/pykaldi/tools/kaldi python setup.py install

EXPOSE 6666