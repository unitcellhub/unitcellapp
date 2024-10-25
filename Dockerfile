# Load in the baseline miniconda environment
FROM continuumio/miniconda3

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]


# Copy in relevant files
# Note: The Heroku webservice clears the /tmp folder, so that isn't a good
# location to store files.
RUN mkdir -p /unitcellapp
COPY requirements.txt /unitcellapp/
COPY docker/ /unitcellapp/docker/

# Initialize conda with bash, which is required prior to the use of conda
RUN conda init bash

# Create a new python environment
RUN conda env create -f /unitcellapp/docker/environment.yml

# Load in the files that are likely to change after the environment
# setup.
COPY dashboard/ /unitcellapp/dashboard/
WORKDIR /unitcellapp/

# Create a new user. This isn't required, but is useful to add in to verify
# that the container runs as a non-root user.
RUN useradd -m unitcellapp
USER unitcellapp

# Run unitcellapp
# Note: Heroku runs as an arbitrary user, which causes issues with the default
# setup of the environment and thus requires some redefinitions. 
CMD conda init 1> /dev/null \
    && source ~/.bashrc \
    && conda activate unitcellapp \
    && gunicorn --config /unitcellapp/docker/gunicorn.config.py --bind 0.0.0.0:$PORT dashboard.index:server
