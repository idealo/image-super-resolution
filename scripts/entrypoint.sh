#!/usr/bin/env bash
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
main_dir="$(dirname $script_dir)"

echo "Hello. This is your favorite ISR assistant."
echo ""

data_flag="none"
train_test_flag="none"
weights_flag="none"

for var in "$@"; do

  if [ $var = "train" ]; then
    if [ $train_test_flag = "none" ]; then
      echo "Training session."
      train_test_flag="--train"
    else
      echo "Select only one flag between 'test' and 'train'."
    fi
  elif [ $var = "test" ]; then
    if [ $train_test_flag = "none" ]; then
      echo "Prediction."
      train_test_flag="--test"
    else
      echo "Select only one flag between 'test' and 'train'."
    fi
  fi

  if [ $var = "div2k" ]; then
    data_flag="--div2k"
    echo "Training on DIV2K dataset."
  elif [ $var = "custom-data" ]; then
    echo "Training on custom data."
    data_flag="--custom-data"
  fi

  if [ $var = "pre-trained" ]; then
    echo "Using pre-trained weights on DIV2K dataset."
    weights_flag="--pre-trained"
  fi

done

if [ $train_test_flag = "--train" ] && [ $data_flag = "none" ]; then
  echo "Specify a training dataset."
  exit
fi

if [ $data_flag = "none" ]; then
  data_flag=""
fi
if [ $weights_flag = "none" ]; then
  weights_flag=""
fi

if [ $train_test_flag = "none" ]; then
  echo "Missing flag. Select one and only one between 'test' and 'train'."
else
  python3 $main_dir/src/run.py $train_test_flag $data_flag $weights_flag
fi
