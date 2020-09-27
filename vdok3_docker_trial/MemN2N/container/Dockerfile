
# Build an image that can do training and inference in SageMaker

FROM continuumio/anaconda3:4.4.0

RUN conda install -c anaconda setuptools

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         libgl1-mesa-glx \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install janome==0.3.10 && \
    pip install tensorflow==1.14.0 --user && \
    pip install ml_metrics==0.1.4 && \
    pip install --upgrade scipy==1.1.0 && \
    pip uninstall --yes numpy && \
    pip install numpy==1.16.4 && \
    pip uninstall --yes gast && \
    pip install gast==0.2.2 && \
    pip install flask && \
    pip install -U gevent==1.4.0 --ignore-installed && \
    pip install gunicorn

RUN python -m nltk.downloader book

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

RUN git clone https://github.com/mack-the-psych/plimac3.git
RUN git clone https://github.com/mack-the-psych/vdok3.git

RUN echo "/opt/program/plimac3/Lib" > /opt/conda/lib/python3.6/site-packages/plimac-custom.pth
RUN echo "/opt/program/plimac3/Tools" >> /opt/conda/lib/python3.6/site-packages/plimac-custom.pth
RUN echo "/opt/program/vdok3/train" >> /opt/conda/lib/python3.6/site-packages/plimac-custom.pth

# WORKDIR /opt/program/plimac3/Resource/OANC
# RUN python compile_shelve.py

# WORKDIR /opt/program/plimac3/Resource/BCCWJ_frequencylist_suw_ver1_0
# RUN python compile_shelve.py

COPY vdok3_sage /opt/program
WORKDIR /opt/program
