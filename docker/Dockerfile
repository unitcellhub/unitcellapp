
# This is docker file is not intented to be run independently; rather, it is
# intended to be used in multilevel docker files. The primary reason for
# building this docker file separately is to isolate the baseline python
# environment from the overall ldt docker build, which can be hard to degub when
# waiting for all of the required python libraries to build (even though they
# are relatively static from a development standpoint.)

# Load in the baseline miniconda environment
FROM continuumio/miniconda3

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Copy in relevant files from the binder folder
COPY requirements.txt /tmp/
COPY binder/* /tmp/binder/
WORKDIR /tmp/binder

# Install required linux tools
RUN apt-get update && apt-get install `cat apt.txt`

# Initialize conda with bash, which is required prior to the use of conda
RUN conda init bash

# Create a new python environment
RUN conda env create -f environment.yml

# Add conda activate to the bashrc so that the new environment is activated
# prior to further RUN commands
RUN echo "conda activate unitcellapp" >> ~/.bashrc

# Run the post build setup
WORKDIR /tmp
RUN ./postBuild
