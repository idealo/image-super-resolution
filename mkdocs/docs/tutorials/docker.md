# Using ISR with Docker
## Setup

1. Install [Docker](https://docs.docker.com/install/)

2. Clone our repository, get the sample weights from [git lfs](https://git-lfs.github.com/) and cd into it:
```
git clone https://github.com/idealo/image-super-resolution
git lfs pull
cd image-super-resolution
```

3. Build docker image for local usage `docker build -t isr . -f Dockerfile.cpu`

In order to train remotely on **AWS EC2** with GPU

4. Install [Docker Machine](https://docs.docker.com/machine/install-machine/)

5. Install [AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/installing.html)

6. Set up an EC2 instance for training with GPU support. You can follow our [nvidia-docker-keras](https://github.com/idealo/nvidia-docker-keras) project to get started

## Prediction
Place your images (`png`, `jpg`) under `data/input/<data name>`, the results will be saved under `/data/output/<data name>/<model>/<training setting>`.

NOTE: make sure that your images only have 3 layers (the `png` format allows for 4).

Check the configuration file `config.yml` for more information on parameters and default folders.

The `-d` flag in the run command will tell the program to load the weights specified in `config.yml`. It is possible though to iteratively select any option from the command line.

### Predict locally
From the main folder run
```
docker run -v $(pwd)/data/:/home/isr/data -v $(pwd)/weights/:/home/isr/weights -v $(pwd)/isr/config.yml:/home/isr/config.yml -it isr -p -d -c config.yml
```
### Predict on AWS with nvidia-docker
From the remote machine run (using our [DockerHub image](https://hub.docker.com/r/idealo/image-super-resolution-gpu/))
```
sudo nvidia-docker run -v $(pwd)/isr/data/:/home/isr/data -v $(pwd)/isr/weights/:/home/isr/weights -v $(pwd)/isr/config.yml:/home/isr/config.yml -it idealo/image-super-resolution-gpu -p -d -c config.yml
```

## Training
Train either locally with (or without) Docker, or on the cloud with `nvidia-docker` and AWS.

Add you training set, including training and validation Low Res and High Res folders, under `training_sets` in `config.yml`.

### Train on AWS with GPU support using nvidia-docker
To train with the default settings set in `config.yml` follow these steps:
1. From the main folder run ```bash scripts/setup.sh -m <name-of-ec2-instance> -b -i -u -d <data_name>```.
2. ssh into the machine ```docker-machine ssh <name-of-ec2-instance>```
3. Run training with ```sudo nvidia-docker run -v $(pwd)/isr/data/:/home/isr/data -v $(pwd)/isr/logs/:/home/isr/logs -v $(pwd)/isr/weights/:/home/isr/weights -v $(pwd)/isr/config.yml:/home/isr/config.yml -it isr -t -d -c config.yml```

`<data_name>` is the name of the folder containing your dataset. It must be under `./data/<data_name>`.


#### Tensorboard
The log folder is mounted on the docker image. Open another EC2 terminal and run
```
tensorboard --logdir /home/ubuntu/isr/logs
```
and locally
```
docker-machine ssh <name-of-ec2-instance> -N -L 6006:localhost:6006
```

#### Notes
A few helpful details
- <b>DO NOT</b> include a Tensorflow version in ```requirements.txt``` as it would interfere with the version installed in the Tensorflow docker image
- <b>DO NOT</b> use ```Ubuntu Server 18.04 LTS``` AMI. Use the ```Ubuntu Server 16.04 LTS``` AMI instead

### Train locally
#### Train locally with docker
From the main project folder run
```
docker run -v $(pwd)/data/:/home/isr/data -v $(pwd)/logs/:/home/isr/logs -v $(pwd)/weights/:/home/isr/weights -v $(pwd)/isr/config.yml:/home/isr/config.yml -it isr -t -d -c config.yml
```
