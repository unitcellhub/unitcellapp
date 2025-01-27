# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# To make the images visable on github, the image needs to be labeled
LABEL org.opencontainers.image.source="https://github.com/unitcellhub/unitcellapp"

# Since some of the packages don't have pip wheels, they need to be
# installed from git. To do so, we need to install git.
# Additionally, some of the visualization libraries require libgl
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y git libgl1-mesa-glx libxrender1

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
# Note, ideally you would run the uv command with the --no-dev option
# as well; however, there is a bug in the Dash pyproject that doesn't
# include all of the necessary packages in the no-dev dependencies.
# So, for now, all of the dev dependencies are included.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project 

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Create a new user. This isn't required, but is useful to add in to verify
# that the container runs as a non-root user.
RUN useradd -m unitcellapp
RUN touch /app/src/config.json && chown unitcellapp:unitcellapp /app/src/config.json
USER unitcellapp

# Run the application by using gunicorn by default
# CMD ["gunicorn", "--config", "docker/gunicorn.config.py", "unitcellapp.index:server"]
CMD ["python", "-c", "from unitcellapp.index import production; production()"]
