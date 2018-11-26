FROM tensorflow/tensorflow:latest-py3

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

ADD /src/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

ENV TENSOR_HOME /home/isr
WORKDIR $TENSOR_HOME

COPY src ./src
COPY scripts ./scripts
ADD config.json ./config.json
COPY weights ./weights


ENV PYTHONPATH='./src/:$PYTHONPATH'
ENTRYPOINT ["sh", "./scripts/entrypoint.sh"]
