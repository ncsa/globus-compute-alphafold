ARG CUDA=11.1.1
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
  build-essential \
  cmake \
  cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
  git \
  hmmer \
  kalign \
  tzdata \
  wget \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get autoremove -y \
  && apt-get clean


# HHsuite static AVX2 build
RUN mkdir /tmp/hh-suite \
    && wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz -P /tmp/hh-suite \
    && mkdir /opt/hhsuite \
    && tar -xf /tmp/hh-suite/hhsuite-3.3.0-AVX2-Linux.tar.gz -C /opt/hhsuite \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && rm -rf /tmp/hh-suite


# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh

# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda install -qy conda==4.13.0 \
    && conda install -y -c conda-forge openmm=7.7.0 cudatoolkit==${CUDA_VERSION} pdbfixer \
      pip \
      python=3.8 \
      && conda clean --all --force-pkgs-dirs --yes


RUN pip install git+https://github.com/ritwik-deshpande/alphafold.git
RUN pip install globus-compute-sdk==2.2.0
RUN pip install globus-compute-endpoint==2.2.0

RUN wget -q -P /opt/conda/lib/python3.8/site-packages/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.
RUN bash ldconfig