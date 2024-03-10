# Use an NVIDIA CUDA image as the base
FROM nvidia/cuda:12.3.1-devel-ubi8

ENV LANG C.UTF-8

RUN dnf update -y && \
    dnf install -y curl gcc gcc-c++ make && \
    dnf clean all

RUN curl -O https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar xvf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make altinstall

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py

RUN python3.10 -m pip install setuptools_rust

RUN python3.10 -m pip install poetry

ENV PATH="/root/.local/bin:${PATH}"
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy the project files into the container
WORKDIR /Malware-Detection
COPY . .

# Install the Python dependencies
RUN poetry install

# Expose the port the app runs on
EXPOSE 1234

# Command to run the application
CMD ["poetry", "run", "app.py"]