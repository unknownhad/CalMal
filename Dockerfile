FROM nvidia/cuda:12.3.1-devel-ubi8
ENV LANG C.UTF-8

# Install system dependencies, omitting htop and hdf5-devel
RUN yum update -y && \
    yum install -y python3.11 python3-devel gcc gcc-c++ curl findutils \
    nano graphviz && \  
    yum clean all && rm -rf /var/cache/yum

# Install pip and Poetry
RUN curl -sSL https://install.python-poetry.org | python3.11 -

# Set the working directory in the container
WORKDIR /Malware-Detection

# Copy the Python dependency files to the container
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
# RUN poetry shell && poetry install

# Copy the rest of your application code
COPY . .