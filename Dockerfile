# # Load in the baseline miniconda environment
# FROM continuumio/miniconda3
#
# # Generate quirements file using
# # uv pip compile pyproject.toml -o requirements.txt
#
# # Make RUN commands use `bash --login`:
# SHELL ["/bin/bash", "--login", "-c"]
#
#
# # Copy in relevant files
# # Note: The Heroku webservice clears the /tmp folder, so that isn't a good
# # location to store files.
# RUN mkdir -p /unitcellapp
# COPY requirements.txt /unitcellapp/
# COPY docker/ /unitcellapp/docker/
#
# # Initialize conda with bash, which is required prior to the use of conda
# RUN conda init bash
#
# # Create a new python environment
# RUN --mount=type=cache,target=/root/.cache/conda \
#     conda env create -f /unitcellapp/docker/environment.yml
#
# # Load in the files that are likely to change after the environment
# # setup.
# COPY src/unitcellapp/ /unitcellapp/unitcellapp/
# WORKDIR /unitcellapp/
#
# # Create a new user. This isn't required, but is useful to add in to verify
# # that the container runs as a non-root user.
# RUN useradd -m unitcellapp
# USER unitcellapp
#
# # Run unitcellapp
# # Note: Heroku runs as an arbitrary user, which causes issues with the default
# # setup of the environment and thus requires some redefinitions. 
# CMD conda init 1> /dev/null \
#     && source ~/.bashrc \
#     && conda activate unitcellapp \
#     && gunicorn --config /unitcellapp/docker/gunicorn.config.py --bind 0.0.0.0:$PORT unitcellapp.index:server
#
# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Since some of the packages don't have pip wheels, they need to be
# installed from git. To do so, we need to install git.
# Additionally, some of the visualization libraries require libgl
RUN apt-get update && apt-get install -y git libgl1-mesa-glx libxrender1

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Normally the build version number is dynamically determined from
# the current git hash/tag. However, the git respository isn't exposed.
# For simplicity, the version is just hard set to a meaningless value.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=LICENSE,target=LICENSE \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --no-install-project --no-dev 

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
# ADD . /app
COPY ./src /app/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Create a new user. This isn't required, but is useful to add in to verify
# that the container runs as a non-root user.
RUN useradd -m unitcellapp
USER unitcellapp

# Run the application by using gunicorn by default
CMD ["gunicorn", "--config", "docker/gunicorn.config.py", "unitcellapp.index:server"]

