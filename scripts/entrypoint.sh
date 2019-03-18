#!/usr/bin/env bash
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
main_dir="$(dirname $script_dir)"

echo "Hello. This is your favorite ISR assistant."
echo ""

train_flag=""
prediction_flag=""
default_flag=""
config_file=""

print_usage() {
  printf "Usage:"
  printf "-c : configuration file path "
  printf "-t : training session "
  printf "-p : prediction session "
  printf "-d : load default values defined in config file "
}

while getopts 'ptdc:' flag; do
  case "${flag}" in
    c) config_file="--config ${OPTARG}" ;;
    t) train_flag="--training" ;;
    p) prediction_flag="--prediction" ;;
    d) default_flag="--default" ;;
    *) print_usage
       exit 1 ;;
  esac
done


python3 $main_dir/ISR/assistant.py $train_flag $prediction_flag $default_flag $config_file
