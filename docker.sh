#!/bin/bash
# Build new docker image of the Lattice Design Tool and a pointing Dockerfile

# Use the supplied command line option for the git respository if supplied.
# This is particularly useful if you already have the LatticeDesignTool locally
# as pulling down the database cache files can take a while.
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--local)
      GITREPOSITORY="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

# Inspired by https://github.com/jupyterhub/repo2docker-action/blob/master/create_docker_image.sh

# General setup script with relevant variables and references.
# Intented to be sourced in other scripts.

# NOTE: github cli must be installed and authenticated

# Set default repository if it hasn't been defined yet
echo $GITREPOSITORY
if [ -n "${GITREPOSITORY}" ]; then
  echo "Git repository already set to ${GITREPOSITORY}"
else
  GITREPOSITORY="https://github.com/unitcellhub/unitcellapp.git"
  echo "Falling back to default git repository ${GITREPOSITORY}"
fi

# Log into docker repository
docker login --username="unitcellhub" dockerhub.com

BRANCH=`git rev-parse --abbrev-ref HEAD`
GITHUB_SHA=`git ls-remote ${GITREPOSITORY} ${BRANCH}`
INPUT_IMAGE_NAME="dockerhub.com/unitcellhub/unitcellapp"
shortSHA=$(echo "${GITHUB_SHA}" | cut -c1-12)
SHA_NAME="${INPUT_IMAGE_NAME}:${shortSHA}"
# DOCKERFILE="binder/Dockerfile"

# Build the docker images with repo2docker
repo2docker --no-run --debug --ref ${BRANCH} --image-name ${SHA_NAME} --cache-from "${INPUT_IMAGE_NAME}:develop" --cache-from "${INPUT_IMAGE_NAME}:release" ${GITREPOSITORY}
docker tag ${SHA_NAME} ${INPUT_IMAGE_NAME}:${BRANCH}

# # Update the Dockerfile to point to the right image. Note, this is required
# # so that mybinder sees that the file has changed and should pull a new image.
# echo "FROM ${SHA_NAME}" > ${DOCKERFILE}
# git add ${DOCKERFILE}
