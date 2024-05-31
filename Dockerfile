# This Dockerfile can be individually configured as seen in https://docs.docker.com/engine/reference/builder/
# all neccessary functionalities of the image and container generated with this file can be found in the synapse wiki

# Either use a base image like pytorch "FROM pytorch/pytorch" (installed with "docker pull pytorch/pytorch") 
# or tensorflow "FROM tensorflow/tensorflow" (installed with "docker pull tensorflow/tensorflow")
# or use miniconda3 and install the used ML-Framework as a dependency
# FROM continuumio/miniconda3
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser
ARG DEBIAN_FRONTEND=noninteractive

# keep index up to date
# RUN apt-get --allow-releaseinfo-change update

# install pip
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip && apt-get install openslide-tools -y
# copy requirements files to container
# pip_requirements.txt can be generated with "pip freeze >> pip_requirements.txt"
COPY pip_requirements.txt /workspace/pip_requirements.txt
# conda_requirements.txt can be generated with "conda list -e >> conda_requirements.txt"
# remove this line if conda is not used

# install the requirements
# RUN pip install -r /workspace/pip_another_requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /workspace/pip_requirements.txt
# remove this line if conda is not used
# RUN conda install -n base --file /workspace/conda_requirements.txt

# install additional requirements
# remove this line if openCV2 is not used
RUN apt-get install libglib2.0-0

USER appuser
# copy the folder of the Dockerfile and all its subfolders to the workspace folder and its subfolders in the container
ADD . /workspace/
RUN chmod +x /workspace/e2e_inference_online.sh
ENTRYPOINT ["bash", "./workspace/e2e_inference_online.sh"]
