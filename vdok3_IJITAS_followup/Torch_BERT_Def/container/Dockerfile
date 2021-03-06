FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
         curl \
         git \
         unzip \
         bzip2 \
         libgl1-mesa-glx \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pyenv/pyenv.git .pyenv

WORKDIR /
ENV HOME  /
ENV PYENV_ROOT /.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install anaconda3-5.0.0
RUN pyenv global anaconda3-5.0.0
RUN pyenv rehash

# RUN python -m nltk.downloader book

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV PYTHONIOENCODING=utf-8

WORKDIR /opt/program
RUN git clone https://github.com/mack-the-psych/plimac3.git

RUN echo "/opt/program/plimac3/Lib" > /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/plimac-custom.pth
RUN echo "/opt/program/plimac3/Tools" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/plimac-custom.pth

RUN conda install -c anaconda setuptools
RUN pip install --upgrade pip && \
    pip install tensorflow-gpu==1.14.0 --user && \
    pip install ml_metrics==0.1.4 && \
    pip install --upgrade scipy==1.1.0 && \
    conda clean --all && \
    conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch && \
    pip install torchtext==0.4.0 && \
    pip install attrdict==2.0.1 && \
    pip uninstall --yes numpy && \
    pip install numpy==1.16.4 && \
    pip uninstall --yes gast && \
    pip install gast==0.2.2 && \
    pip install -U gevent==1.4.0 --ignore-installed && \
    pip install gunicorn
    
WORKDIR /opt/program
RUN git clone https://github.com/mack-the-psych/vdok3.git

# RUN echo "/workdir/vdok3/prep" > /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
# RUN echo "/workdir/vdok3/extract" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
# RUN echo "/workdir/vdok3/process" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
# RUN echo "/workdir/vdok3/reorganize" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
# RUN echo "/workdir/vdok3/train" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
# RUN echo "/workdir/vdok3/train/pytorch_advanced/nlp_sentiment_bert" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth

RUN echo "/opt/program/vdok3/prep" > /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/opt/program/vdok3/extract" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/opt/program/vdok3/process" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/opt/program/vdok3/reorganize" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/opt/program/vdok3/train" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth
RUN echo "/opt/program/vdok3/train/pytorch_advanced/nlp_sentiment_bert" >> /.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/vdok3-custom.pth

WORKDIR /opt/program/vdok3/train/pytorch_advanced/nlp_sentiment_bert
RUN python make_folders_and_data_downloads.py

COPY vdok3_sage /opt/program
WORKDIR /opt/program
