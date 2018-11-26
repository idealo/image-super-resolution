script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
main_dir="$(dirname $script_dir)"

build="false"
no_d="false"
install="false"

for var in "$@"; do
  if [ $var = "build" ]; then
    build="true"
  elif [ $var = "no_d" ]; then
    no_d="true"
  elif [ $var = "install" ]; then
    install="true"
  fi
done

if [ $build = "true" ]; then
  echo "Creating Docker image"
  sudo nvidia-docker build -f $main_dir/Dockerfile.gpu -t isr $main_dir
fi

if [ $install = "true" ]; then
  echo "Installing unzip"
  sudo apt install unzip python-pip
  echo "Installing tensorboard"
  pip install tensorflow
fi

if [ $no_d = "false" ]; then
  echo "Downloading Dataset (this might take a while)"
  python3 $script_dir/getDIV2K.py
fi
