machine_name=$1
if [ -z $1 ]; then
  echo "Specify machine name"
  exit
fi

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

script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
main_dir="$(dirname $script_dir)"

bash $script_dir/update_aws.sh $machine_name
wait
flag="no"
flag1="no"

if [ $build = "true" ] ; then
  flag="build"
fi

if [ $no_d = "true" ] ; then
  flag1="no_d"
fi
if [ $install = "true" ] ; then
  flag2="install"
fi
docker-machine ssh $machine_name << EOF
  bash isr/scripts/build_n_download.sh $flag $flag1 $flag2
EOF
