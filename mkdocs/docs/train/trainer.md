## class Trainer
Class object to setup and carry the training.

Takes as input a generator that produces SR images. Conditionally, also a discriminator network and a feature extractor to build the components of the perceptual loss. Compiles the model(s) and trains in a GANS fashion if a discriminator is provided, otherwise carries a regular ISR training.
##### Args
* **generator**: Keras model, the super-scaling, or generator, network.

* **discriminator**: Keras model, the discriminator network for the adversarial
    component of the perceptual loss.

* **feature_extractor**: Keras model, feature extractor network for the deep features
    component of perceptual loss function.

* **lr_train_dir**: path to the directory containing the Low-Res images for training.

* **hr_train_dir**: path to the directory containing the High-Res images for training.

* **lr_valid_dir**: path to the directory containing the Low-Res images for validation.

* **hr_valid_dir**: path to the directory containing the High-Res images for validation.

* **learning_rate**: float.

* **loss_weights**: dictionary, use to weigh the components of the loss function.
    Contains 'MSE' for the MSE loss component, and can contain 'discriminator' and 'feat_extr'
    for the discriminator and deep features components respectively.

* **logs_dir**: path to the directory where the tensorboard logs are saved.

* **weights_dir**: path to the directory where the weights are saved.

* **dataname**: string, used to identify what dataset is used for the training session.

* **weights_generator**: path to the pre-trained generator's weights, for transfer learning.

* **weights_discriminator**: path to the pre-trained discriminator's weights, for transfer learning.

* **n_validation**: integer, number of validation samples used at training from the validation set.

* **T**: 0 < float <1, determines the 'flatness' threshold level for the training patches.
    See the TrainerHelper class for more details.

* **lr_decay_frequency**: integer, every how many epochs the learning rate is reduced.

* **lr_decay_factor**: 0 < float <1, learning rate reduction multiplicative factor.

* **ods**: 

* **train**: combines the networks and triggers training with the specified settings.

### \_\_init\_\_
```python
def __init__(generator, discriminator, feature_extractor, lr_train_dir, hr_train_dir, lr_valid_dir, hr_valid_dir, learning_rate, loss_weights, logs_dir, weights_dir, dataname, weights_generator, weights_discriminator, n_validation, T, lr_decay_frequency, lr_decay_factor)
```

### train
```python
def train(epochs, steps_per_epoch, batch_size)
```
Carries on the training for the given number of epochs. Sends the losses to Tensorboard.



