## class Trainer
Class object to setup and carry the training.

Takes as input a generator that produces SR images. Conditionally, also a discriminator network and a feature extractor. Compiles the model(s) and trains in a GANS fashion if a discriminator is provided, otherwise carries a regular ISR training.
### \_\_init\_\_
```python
def __init__(generator, discriminator, feature_extractor, lr_train_dir, hr_train_dir, lr_valid_dir, hr_valid_dir, learning_rate, loss_weights, logs_dir, weights_dir, dataname, weights_generator, weights_discriminator, n_validation, T, lr_decay_frequency, lr_decay_factor)
```

### train
```python
def train(epochs, steps_per_epoch, batch_size)
```
Carries on the training for the given number of epochs. Sends the losses to Tensorboard.



