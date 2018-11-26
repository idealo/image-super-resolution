machine_name=$1
if [ -z $1 ]; then
  echo "Specify machine name"
  exit
fi

createmachine="false"
build="false"
no_d="false"
install="false"

for var in "$@"; do
  if [ $var = "create-machine" ]; then
    createmachine="true"
  elif [ $var = "build" ]; then
    build="true"
  elif [ $var = "no_d" ]; then
    no_d="true"
  elif [ $var = "install" ]; then
    install="true"
  fi
done

if [ $createmachine = "true" ]; then
  docker-machine create --driver amazonec2 \
--amazonec2-region eu-central-1 --amazonec2-zone=a \
--amazonec2-ami ami-cbccf120 \
--amazonec2-instance-type p2.xlarge \
--amazonec2-vpc-id vpc-f3bcb89b \
--amazonec2-root-size 95 \
$machine_name
fi
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
