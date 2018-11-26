# Image Super-Resolution with Keras and Docker

<img src="figures/butterfly.png">

The goal of this project is to upscale low resolution images by a factor of 2. To achieve this we used the CNN Residual Dense Network described in [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf) (Zhang et al. 2018). <br> We wrote a Keras implementation of the network and set up a [Docker](https://docs.docker.com/install/) image to carry training and testing, locally or on the cloud on [AWS EC2 instances](https://github.com/idealo/nvidia-docker-keras) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), with only a few commands.

## Contents
- [Sample Results](#sample-results)
- [Getting Started](#getting-started)
- [Predict](#predict)
- [Train](#train)
- [Unit Testing](#unit-testing)
- [Additional Information](#additional-information)
- [Mantainers](#mantainers)
- [License](#copyright)

## Sample Results

Below we show the original low resolution image (center), the super scaled output of the network (center) and the result of the baseline scaling obtained with GIMP bicubic scaling (right).

<br>
<img src="figures/butterfly_comparison_SR_baseline.png">

<br>
<img src="figures/basket_comparison_SR_baseline.png">

## Getting Started

1. Install [Docker](https://docs.docker.com/install/)

2. Build docker image `sudo docker build -t isr . -f Dockerfile.cpu`

In order to train remotely  **AWS EC2** with GPU

3. Install [Docker Machine](https://docs.docker.com/machine/install-machine/)

4. Install [AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/installing.html)

5. Set up an [EC2 instance](https://github.com/idealo/nvidia-docker-keras)

6. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on the EC2 instance

## Predict
The default folder for tests is ```data/test/custom_test```.
The results are saved under ```data/results/custom_results```.

Check `config.json` for more information on parameters and default folders.

### Predict locally
From the main folder run
```
docker run -v $(pwd)/data/:/home/isr/data -it isr test pre-trained
```
### Predict on AWS with nvidia-docker
From the remote machine run (using our DockerHub image)
```
sudo nvidia-docker run -v $(pwd)/isr/data/:/home/isr/data -it idealo/image-super-resolution test pre-trained
```

## 4. Train
Train either locally with (or without) Docker, or on the cloud with `nvidia-docker` and AWS.

### Train on AWS with GPU support using nvidia-docker
1. Run ```setup.sh <name-of-ec2-instance> build install``` from the scripts folder. Add the `no-d` flag if you want to use a custom dataset.
2. ssh into the machine ```docker-machine ssh <name-of-ec2-instance>```
3. Run training with ```sudo nvidia-docker run -v $(pwd)/isr/data/:/home/isr/data -v $(pwd)/isr/logs/:/home/isr/logs -it isr train div2k```


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
- <b>DO NOT</b> include a tensorflow version in  ```requirements.txt``` as it would interfere with the version installed in the tensorflow docker image.
- <b>DO NOT</b> use ```Ubuntu Server 18.04 LTS``` AMI instead of ```Ubuntu Server 16.04 LTS``` AMI.

### Train locally
#### Download dataset
Download the DIV2K dataset by running `python scripts/getDIV2K.py`. <br>
If you want to use a custom dataset, adjust `config.json` accordingly and replace the `div2k` flag with `custom-data`.
#### Train locally with docker
```
docker run -it isr train div2k
```


## Unit Testing
From ```./src``` folder run
```
python -m pytest -vs --disable-pytest-warnings tests
```


## Additional Information
### Network architecture
The main parameters of the architecture structure are:
- D - number of Residual Dense Blocks (RDB)
- C - number of Convolutional Layers stacked inside a RDB;
- G - number of feature maps of each Conv Layer inside the RDBs;
- G0 - number of feature maps for convolutions outside of RDBs and of each RBD output.


<img src="figures/RDN.png" width="600">
<br>

<img src="figures/RDB.png" width="600">

### Pre-trained weights
Pre-trained weights on [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K) are available under ```weights/sample_weights```. <br>
The model was trained using  ```--D=20 --G=64 --C=6``` as parameters (see architecture for details) for 86 epochs of 1000 batches of 8 32x32 augmented patches taken from LR images.

## Maintainers
* Francesco Cardinale, github: [cfrancesco](https://github.com/cfrancesco)
* Zubin John, github: [valiantone](https://github.com/valiantone)


## Copyright

See [LICENSE](LICENSE) for details.
