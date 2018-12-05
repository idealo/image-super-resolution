#!/usr/bin/env bash
local_script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
local_main_dir="$(dirname "$local_script_dir")"
machine_name=$1
aws_main_dir=/home/ubuntu/isr
build="false"
install="false"
update="false"
weights="true"
data="false"


if [ -z $1 ]; then
  echo "Error: Specify machine name"
  exit
fi


for var in "$@"; do
  if [ $var = "build" ]; then
    build="true"
  elif [ $var = "install" ]; then
    install="true"
  elif [ $var = "div2k" ]; then
    data="DIV2K"
  elif [ $var = "custom-data" ]; then
    data="custom"
  elif [ $var = "update" ]; then
    update="true"
  elif [ $var = "no_weights" ]; then
    weights="false"
  fi
done

if [ $update = "true" ]; then
  echo " >>> Copying local source files to remote machine."
  docker-machine scp -r $local_main_dir/src $machine_name:$aws_main_dir
  docker-machine scp -r $local_main_dir/config.json $machine_name:$aws_main_dir
  docker-machine scp -r $local_main_dir/Dockerfile.gpu $machine_name:$aws_main_dir
  docker-machine scp -r $local_main_dir/.dockerignore $machine_name:$aws_main_dir
  docker-machine scp -r $local_main_dir/scripts/entrypoint.sh $machine_name:$aws_main_dir/scripts

  if [ $weights = "true" ]; then
    docker-machine scp $local_main_dir/weights/sample_weights/DIV2K_E086_X2_D20G64C6.hdf5 \
  $machine_name:$aws_main_dir/weights/sample_weights/
  fi
fi

if ! [ $data = "false" ]; then
  echo " >>> Copying local data folder to remote machine. This will take some time (output is suppressed)"
  docker-machine scp -r -q $local_main_dir/data/$data $machine_name:$aws_main_dir/data/$data
fi

if [ $build = "true" ]; then
  echo " >>> Connecting to the remote machine."
  docker-machine ssh $machine_name << EOF
    echo " >>> Creating Docker image"
    sudo nvidia-docker build -f $aws_main_dir/Dockerfile.gpu -t isr $aws_main_dir
EOF
fi

if [ $install = "true" ]; then
  echo " >>> Connecting to the remote machine."
  docker-machine ssh $machine_name << EOF
  echo "Installing unzip"
  sudo apt -y install unzip
  echo "Updating pip"
  pip install --upgrade pip
  echo "Installing tensorboard"
  pip install tensorflow
EOF
fi
