#!/usr/bin/env bash
local_script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
local_main_dir="$(dirname "$local_script_dir")"
aws_main_dir=/home/ubuntu/isr

machine_name="false"
weights="false"
install="false"
update="false"
build="false"
data="false"

print_usage() {
  printf "Usage:"
  printf "-w : upload pre-trained weights "
  printf "-b : automatically build docker image "
  printf "-i : install required packages "
  printf "-u : update code base "
  printf "-d <data_name> : upload a dataset from the data folder "
  printf "-m <instance_name> : the name of the ec2 instance "
}

while getopts 'm:biuwd:' flag; do
  case "${flag}" in
    m) machine_name="${OPTARG}" ;;
    b) build="true" ;;
    i) install="true" ;;
    u) update="true" ;;
    w) weights="true" ;;
    d) data="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

if [ $machine_name = "false" ]; then
  echo "Error: Specify machine name"
  exit
fi

if [ $update = "true" ]; then
  docker-machine ssh $machine_name << EOF
  mkdir -p $aws_main_dir/ISR
  mkdir $aws_main_dir/scripts
EOF

  echo " >>> Copying local source files to remote machine."
  docker-machine scp -r $local_main_dir/ISR $machine_name:$aws_main_dir
  docker-machine scp -r $local_main_dir/config.yml $machine_name:$aws_main_dir
  docker-machine scp -r $local_main_dir/setup.py $machine_name:$aws_main_dir
  docker-machine scp $local_main_dir/Dockerfile.gpu $machine_name:$aws_main_dir
  docker-machine scp $local_main_dir/.dockerignore $machine_name:$aws_main_dir
  docker-machine scp $local_main_dir/scripts/entrypoint.sh $machine_name:$aws_main_dir/scripts/
 fi

if [ $weights = "true" ]; then
  docker-machine ssh $machine_name << EOF
  mkdir -p $aws_main_dir/weights/sample_weights/rdn-C6-D20-G64-G064-x2/PSNR-driven/
  mkdir -p $aws_main_dir/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/
  mkdir -p $aws_main_dir/weights/sample_weights/rdn-C3-D10-G64-G064-x2/PSNR-driven/
EOF
  docker-machine scp $local_main_dir/weights/sample_weights/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5 \
$machine_name:$aws_main_dir/weights/sample_weights/rdn-C6-D20-G64-G064-x2/PSNR-driven/
  docker-machine scp $local_main_dir/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5 \
$machine_name:$aws_main_dir/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/
  docker-machine scp $local_main_dir/weights/sample_weights/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5 \
$machine_name:$aws_main_dir/weights/sample_weights/rdn-C3-D10-G64-G064-x2/PSNR-driven/
fi


if ! [ $data = "false" ]; then
  docker-machine ssh $machine_name << EOF
  mkdir -p $aws_main_dir/data/
EOF
  echo " >>> Copying local data folder to remote machine. This will take some time (output is suppressed)"
  docker-machine scp -r -q $local_main_dir/data/$data $machine_name:$aws_main_dir/data
fi

if [ $build = "true" ]; then
  echo " >>> Connecting to the remote machine."
  docker-machine ssh $machine_name << EOF
    echo " >>> Creating Docker image"
    sudo nvidia-docker build -f $aws_main_dir/Dockerfile.gpu -t isr $aws_main_dir --rm
EOF
fi

if [ $install = "true" ]; then
  echo " >>> Connecting to the remote machine."
  docker-machine ssh $machine_name << EOF
  echo " >>> Updating pip"
  python3 -m pip install --upgrade pip
  echo " >>> Installing tensorboard"
  python3 -m pip install tensorflow --user
EOF
fi
