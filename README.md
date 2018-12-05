# Image Super-Resolution

<img src="figures/butterfly.png">

The goal of this project is to upscale low resolution images (currently only x2 scaling). To achieve this we used the CNN Residual Dense Network described in [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf) (Zhang et al. 2018).

We wrote a Keras implementation of the network and set up a Docker image to carry training and testing. You can train either locally or on the cloud with AWS and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) with only a few commands.

We welcome any kind of contribution. If you wish to contribute, please see the [Contribute](#contribute) section.

## Contents
- [Sample Results](#sample-results)
- [Getting Started](#getting-started)
- [Predict](#predict)
- [Train](#train)
- [Unit Testing](#unit-testing)
- [Additional Information](#additional-information)
- [Contribute](#contribute)
- [Maintainers](#maintainers)
- [License](#copyright)

## Sample Results

Below we show the original low resolution image (centre), the super scaled output of the network (centre) and the result of the baseline scaling obtained with GIMP bicubic scaling (right).

<br>
<img src="figures/butterfly_comparison_SR_baseline.png">

<br>
<img src="figures/basket_comparison_SR_baseline.png">

## Getting Started

1. Install [Docker](https://docs.docker.com/install/)

2. Install [Anaconda](https://www.anaconda.com/)

3. Create the conda environment `conda env create -f src/environment.yml`

4. Build docker image for local usage `docker build -t isr . -f Dockerfile.cpu`

In order to train remotely on **AWS EC2** with GPU

5. Install [Docker Machine](https://docs.docker.com/machine/install-machine/)

6. Install [AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/installing.html)

7. Set up an EC2 instance for training with GPU support. You can follow our [nvidia-docker-keras](https://github.com/idealo/nvidia-docker-keras) project to get started

## Predict
Place your images (`png`, `jpg`) under `data/input`, the results will be saved under `/data/output`.

Check `config.json` for more information on parameters and default folders.

NOTE: make sure that your images only have 3 layers (the `png`  format allows for 4).

### Predict locally
From the main folder run
```
docker run -v $(pwd)/data/:/home/isr/data isr test pre-trained
```
### Predict on AWS with nvidia-docker
From the remote machine run (using our [DockerHub image](https://hub.docker.com/r/idealo/image-super-resolution-gpu/))
```
sudo nvidia-docker run -v $(pwd)/isr/data/:/home/isr/data idealo/image-super-resolution-gpu test pre-trained
```

## Train
Train either locally with (or without) Docker, or on the cloud with `nvidia-docker` and AWS. <br>
Place your training and validation datasets under `data/custom` with the following structure:
- Training low resolution images under `data/custom/lr/train`
- Training high resolution images under `data/custom/hr/train`
- Validation low resolution images under `data/custom/lr/validation`
- Validation high resolution images under `data/custom/hr/validation`

Use the `custom-data` flag in the train and setup commands (replace `<dataset-flag>`) to train on this dataset.

We trained our model on the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K) dataset. If you want to train on this dataset too, you can download it by running `python scripts/getDIV2K.py` from the main folder. This will place the dataset under its default folders (check `config.json` for more details). Use the `div2k` flag for the train command to train on this dataset.

### Train on AWS with GPU support using nvidia-docker
1. From the main folder run ```bash scripts/setup.sh <name-of-ec2-instance> build install update <dataset-flag>```. The available dataset-flags are `div2k` and `custom-data`. Make sure you downloaded the dataset first.
2. ssh into the machine ```docker-machine ssh <name-of-ec2-instance>```
3. Run training with ```sudo nvidia-docker run -v $(pwd)/isr/data/:/home/isr/data -v $(pwd)/isr/logs/:/home/isr/logs -it isr train <dataset-flag>```


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
docker run -v $(pwd)/data/:/home/isr/data -v $(pwd)/logs/:/home/isr/logs -it isr train <dataset-flag>
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
- C - number of convolutional layers stacked inside a RDB
- G - number of feature maps of each convolutional layers inside the RDBs
- G0 - number of feature maps for convolutions outside of RDBs and of each RBD output


<img src="figures/RDN.png" width="600">
<br>

<img src="figures/RDB.png" width="600">

source: [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf)

### Pre-trained weights
Pre-trained model weights on [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K) are available under ```weights/sample_weights```. <br>
The model was trained using ```--D=20 --G=64 --C=6``` as parameters (see architecture for details) for 86 epochs of 1000 batches of 8 augmented patches (32x32) taken from LR images.

## Contribute
We welcome all kinds of contributions, models trained on different datasets, new model architectures and/or hyperparameters combinations that improve the performance of the currently published model.

Will publish the performances of new models in this repository.

## Maintainers
* Francesco Cardinale, github: [cfrancesco](https://github.com/cfrancesco)
* Zubin John, github: [valiantone](https://github.com/valiantone)


## Copyright

See [LICENSE](LICENSE) for details.
