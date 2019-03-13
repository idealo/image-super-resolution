#!/usr/bin/env bash
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
main_dir="$(dirname $script_dir)"

echo "Hello. This is your favorite ISR assistant."
echo ""

train_flag=""
prediction_flag=""
default_flag=""

for var in "$@"; do
  if [ $var = "--training" ]; then
    train_flag=$var
  elif [ $var = "--prediction" ]; then
    prediction_flag=$var
  elif [ $var = "--default" ]; then
    default_flag=$var
  fi
done

python3 $main_dir/ISR/run.py $train_flag $prediction_flag $default_flag
