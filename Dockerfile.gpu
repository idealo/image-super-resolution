FROM tensorflow/tensorflow:1.13.1-gpu-py3

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      screen \
      wget && \
    rm -rf /var/lib/apt/lists/* \
    apt-get upgrade

ENV TENSOR_HOME /home/isr
WORKDIR $TENSOR_HOME

COPY ISR ./ISR
COPY scripts ./scripts
COPY weights ./weights
COPY config.yml ./
COPY setup.py ./

RUN pip install --upgrade pip
RUN pip install -e ".[gpu]" --ignore-installed

ENV PYTHONPATH ./ISR/:$PYTHONPATH
ENTRYPOINT ["sh", "./scripts/entrypoint.sh"]
