# Image Super-Resolution

<img src="figures/butterfly.png">

[![Build Status](https://travis-ci.org/idealo/image-super-resolution.svg?branch=master)](https://travis-ci.org/idealo/image-super-resolution)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://github.com/idealo/image-super-resolution/blob/master/LICENSE)

The goal of this project is to upscale and improve the quality of low resolution images.

Includes the Keras implementations of:

- the super-scaling Residual Dense Network described in [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797) (Zhang et al. 2018);
- the super-scaling Residual in Residual Dense Network described in [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)(Wang et al. 2018)
- a multi-output version of the Keras VGG19 network for deep features extraction used in the perceptual loss;
- a custom discriminator network based on the one described in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (SRGANS, Ledig et al. 2017)

Docker scripts are available to carry training and testing. Also, we provide scripts to facilitate training on the cloud with AWS and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) with only a few commands.

We welcome any kind of contribution. If you wish to contribute, please see the [Contribute](#contribute) section.

## Contents
- [Sample Results](#sample-results)
- [Getting Started](#getting-started)
- [Predict](#predict)
- [Train](#train)
- [Installation](#installation)
- [Unit Testing](#unit-testing)
- [Changelog](#changelog)
- [Additional Information](#additional-information)
- [Contribute](#contribute)
- [Maintainers](#maintainers)
- [License](#copyright)

## Sample Results

The samples use an upscaling factor of two.
The weights used to produced these images are available under `sample_weights` (see [Additional Information](#additional-information)).

The original low resolution image (left), the super scaled output of the network (center) and the result of the baseline scaling obtained with GIMP bicubic scaling (right).

<br>
<img src="figures/butterfly_comparison_SR_baseline.png">

<br>
<img src="figures/basket_comparison_SR_baseline.png">

Below a sample output of the RDN network re-trained on the weights provided in the repo to achieve artefact removal and detail enhancement. For the follow up training session we used a combination of artefact removing strategies and a few forms of a perceptual loss, using combinations of the deep features of the VGG19 and the GAN's discriminator network.

<center>
<figure>
  <img src="figures/ISR-reference.png" alt="my alt text"/>
  <figcaption>Bicubic up-scaling (baseline).</figcaption>
</figure>


<figure>
  <img src="figures/ISR-vanilla-RDN.png" alt="my alt text"/>
  <figcaption>ISR standard.</figcaption>
</figure>

<figure>
  <img src="figures/ISR-gans-vgg.png" alt="my alt text"/>
  <figcaption>ISR with artefact removal and VGG+GAN perceptual loss.</figcaption>
</figure>
</center>

## Getting Started

1. Install [Docker](https://docs.docker.com/install/)

2. Build docker image for local usage `docker build -t isr . -f Dockerfile.cpu`

In order to train remotely on **AWS EC2** with GPU

3. Install [Docker Machine](https://docs.docker.com/machine/install-machine/)

4. Install [AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/installing.html)

5. Set up an EC2 instance for training with GPU support. You can follow our [nvidia-docker-keras](https://github.com/idealo/nvidia-docker-keras) project to get started

## Predict
Place your images (`png`, `jpg`) under `data/input/<data name>`, the results will be saved under `/data/output/<data name>/<model>/<training setting>`.

NOTE: make sure that your images only have 3 layers (the `png` format allows for 4).

Check the configuration file `config.yml` for more information on parameters and default folders.

The `--default` flag in the run command will tell the program to load the weights specified in `config.yml`. It is possible though to iteratively select any option from the command line.

### Predict locally
From the main folder run
```
docker run -v $(pwd)/data/:/home/isr/data -v $(pwd)/weights/:/home/isr/weights -it isr --predict --default --config config.yml
```
### Predict on AWS with nvidia-docker
From the remote machine run (using our [DockerHub image](https://hub.docker.com/r/idealo/image-super-resolution-gpu/))
```
sudo nvidia-docker run -v $(pwd)/isr/data/:/home/isr/data -v $(pwd)/isr/weights/:/home/isr/weights -it idealo/image-super-resolution --predict --default --config config.yml
```

## Train
Train either locally with (or without) Docker, or on the cloud with `nvidia-docker` and AWS.

Add you training set, including training and validation Low Res and High Res folders, under `training_sets` in `config.yml`.

### Train on AWS with GPU support using nvidia-docker
To train with the default settings set in `config.yml` follow these steps:
1. From the main folder run ```bash scripts/setup.sh -m <name-of-ec2-instance> -b -i -u -d <data_name>```.
2. ssh into the machine ```docker-machine ssh <name-of-ec2-instance>```
3. Run training with ```sudo nvidia-docker run -v $(pwd)/isr/data/:/home/isr/data -v $(pwd)/isr/logs/:/home/isr/logs -v $(pwd)/isr/weights/:/home/isr/weights -it isr --training --default --config config.yml```

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
docker run -v $(pwd)/data/:/home/isr/data -v $(pwd)/logs/:/home/isr/logs -v $(pwd)/weights/:/home/isr/weights -it isr --train --default --config config.yml
```

## Installation
There are two ways to install the Image Super-Resolution package:

- Install ISR from PyPI (recommended):
```
pip install ISR
```
- Install ISR from the GitHub source:
```
git clone https://github.com/idealo/image-super-resolution
cd image-super-resolution
python setup.py install
```

## Unit Testing
From the `root` directory run
```
pip install -e .[tests]
pytest -vs --cov=ISR --show-capture=no --disable-pytest-warnings tests/
```

## Changelog
- v2: added deep features from VGG19 and a discriminator for GAN training. Moved all non strictly architecture building operations outside of the model files. The models are combined when needed in the Trainer class. In order to allow for GAN training `fit_generator` function had to be replaced with the more granular `train_on_batch`. Now the project relies on custom data handlers and loggers instead of the custom Keras generator.

## Additional Information
### RDN Network architecture
The main parameters of the architecture structure are:
- D - number of Residual Dense Blocks (RDB)
- C - number of convolutional layers stacked inside a RDB
- G - number of feature maps of each convolutional layers inside the RDBs
- G0 - number of feature maps for convolutions outside of RDBs and of each RBD output

<img src="figures/RDN.png" width="600">
<br>

<img src="figures/RDB.png" width="600">

source: [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)

### RRDN Network architecture
The main parameters of the architecture structure are:
- T - number of Residual in Residual Dense Blocks (RRDB)
- D - number of Residual Dense Blocks (RDB) insider each RRDB
- C - number of convolutional layers stacked inside a RDB
- G - number of feature maps of each convolutional layers inside the RDBs
- G0 - number of feature maps for convolutions outside of RDBs and of each RBD output

<img src="figures/RRDN.jpg" width="600">
<br>

<img src="figures/RRDB.png" width="600">

source: [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

### RDN Pre-trained weights
The weights of the RDN network trained on the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K) are available in ```weights/sample_weights/rdn-C6-D20-G64-G064-x2_div2k-e086.hdf5```. <br>
The model was trained using ```C=6, D=20, G=64, G0=64``` as parameters (see architecture for details) for 86 epochs of 1000 batches of 8 32x32 augmented patches taken from LR images.

The artefact removing and detail enhancing weights obtained with a combination of different training sessions using different datasets and perceptual loss with VGG19 and GAN can be found at `weights/sample_weights/rdn-C6-D20-G64-G064-x2_enhanced-e219.hdf5`

## Contribute
We welcome all kinds of contributions, models trained on different datasets, new model architectures and/or hyperparameters combinations that improve the performance of the currently published model.

Will publish the performances of new models in this repository.

See [CONTRIBUTING](CONTRIBUTING.md) for more details.

## Maintainers
* Francesco Cardinale, github: [cfrancesco](https://github.com/cfrancesco)
* Zubin John, github: [valiantone](https://github.com/valiantone)
* Dat Tran, github: [datitran](https://github.com/datitran)

## Copyright

See [LICENSE](LICENSE) for details.
