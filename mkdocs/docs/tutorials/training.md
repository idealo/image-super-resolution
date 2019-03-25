# ISR Suite: HOW-TO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idealo/image-super-resolution/blob/master/notebooks/ISR_Traininig_Tutorial.ipynb)

## Training

### Get the training data
Get your data to train the model. The div2k dataset linked here is for a scaling factor of 2. Beware of this later when training the model.

```bash
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

mkdir div2k
unzip -q DIV2K_valid_LR_bicubic_X2.zip -d div2k
unzip -q DIV2K_train_LR_bicubic_X2.zip -d div2k
unzip -q DIV2K_train_HR.zip -d div2k
unzip -q DIV2K_valid_HR.zip -d div2k
```

### Create the models
Import the models from the ISR package and create

- a RRDN super scaling network
- a discrmiminator network for GANs training
- a VGG19 feature extractor to train with a perceptual loss function

Carefully select:

- 'x': this is the upscaling factor (2 by default)
- 'layers_to_extract': these are the layers from the VGG19 that will be used in the perceptual loss (leave the default if you're not familiar with it)
- 'lr_patch_size': this is the size of the patches that will be extracted from the LR images and fed to the ISR network during training time

Play around with the other architecture parameters

```python
from ISR.models.rrdn import RRDN
from ISR.models.discriminator import Discriminator
from ISR.models.cut_vgg19 import Cut_VGG19

lr_train_patch_size = 40
layers_to_extract = [5, 9]
scale = 2
hr_train_patch_size = lr_train_patch_size * scale

rrdn  = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)
```

### Give the models to the Trainer
The Trainer object will combine the networks, manage your training data and keep you up-to-date with the training progress through Tensorboard and the command line.

```python
from ISR.trainer.trainer import Trainer
loss_weights = {
  'generator': 0.0,
  'feat_extr': 0.0833,
  'discriminator': 0.01
}
trainer = Trainer(
    generator=rrdn,
    discriminator=discr,
    feature_extractor=f_ext,
    lr_train_dir='div2k/DIV2K_train_LR_bicubic/X2/',
    hr_train_dir='div2k/DIV2K_train_HR/',
    lr_valid_dir='div2k/DIV2K_train_LR_bicubic/X2/',
    hr_valid_dir='div2k/DIV2K_train_HR/',
    loss_weights=loss_weights,
    dataname='div2k',
    logs_dir='./logs',
    weights_dir='./weights',
    weights_generator=None,
    weights_discriminator=None,
    n_validation=40,
    lr_decay_frequency=30,
    lr_decay_factor=0.5,
    T=0.01,
)
```

Choose epoch number, steps and batch size and start training

```python
trainer.train(
    epochs=1,
    steps_per_epoch=20,
    batch_size=4,
)
```
