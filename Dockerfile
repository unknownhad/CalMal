# FROM nvidia/cuda:10.0-cudnn7-runtime
FROM nvidia/cuda:12.3.1-devel-ubi8
ENV LANG C.UTF-8
# RUN apt-get update -qq && apt-get install -qy python3 python3-dev curl libhdf5-dev nano htop
# RUN curl https://bootstrap.pypa.io/get-pip.py | python3 -
# ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu/hdf5/serial/:/usr/local/cuda/lib64"
# RUN pip install --no-cache-dir numpy click requests tensorflow-gpu==1.14.0 keras pandas matplotlib sklearn umap-learn Flask flask-cors anycache graphviz torch torchvision
WORKDIR /Malware-Detection
