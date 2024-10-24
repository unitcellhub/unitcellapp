# Load in the baseline miniconda environment
FROM continuumio/miniconda3

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Copy in relevant files from the binder folder
COPY requirements.txt /tmp/
COPY dashboard/ /tmp/dashboard/
COPY binder/ /tmp/binder/
WORKDIR /tmp/

# Install required linux tools
# RUN apt-get update && apt-get install `cat apt.txt`

# Initialize conda with bash, which is required prior to the use of conda
RUN conda init bash

# Create a new python environment
RUN conda env create -f binder/environment.yml

# Add conda activate to the bashrc so that the new environment is activated
# prior to further RUN commands
RUN echo "conda activate unitcellapp" >> ~/.bashrc

# Run unitcellapp
CMD gunicorn --worker-tmp-dir=/dev/shm dashboard.index:server
