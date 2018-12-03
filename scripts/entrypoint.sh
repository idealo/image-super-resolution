#!/usr/bin/env bash
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
main_dir="$(dirname $script_dir)"

echo "Hello. This is your favorite ISR assistant."
echo ""
error=0
if [ $1 = "train" ]; then
  if [ $2 = "div2k" ]; then
    echo "Training on DIV2K dataset."
    python3 $main_dir/src/run.py --train --div2k
  else
    echo "Training on custom dataset."
    python3 $main_dir/src/run.py --train
  fi
elif [ $1 = "test" ]; then
    if [ $2 = "pre-trained" ]; then
      flag='--pre-trained'
      echo "Using pre-trained weights on DIV2K dataset."
    else
      flag=''
    fi
    echo "Starting prediction."
    echo "Reading images from default folder."
    python3 ./src/run.py --test $flag
else
  echo "ERROR: Invalid run arguments."
fi
/bin/bash
