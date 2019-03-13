## class TrainerHelper
Collection of userful functions to manage training sessions.


### \_\_init\_\_
```python
def __init__(generator, weights_dir, logs_dir, lr_train_dir, feature_extractor, discriminator, dataname, pretrained_weights_path, fallback_save_every_n_epochs)
```

### print\_training\_setting
```python
def print_training_setting()
```
Does what it says.



### on\_epoch\_end
```python
def on_epoch_end(epoch, losses, generator, discriminator, metrics)
```
Manages the operations that are taken at the end of each epoch: metric checks, weight saves, logging.



### epoch\_n\_from\_weights\_name
```python
def epoch_n_from_weights_name(w_name)
```
Extracts the last epoch number from the standardized weights name. Only works with standardized weights names.



### initialize\_training
```python
def initialize_training(object)
```
Function that is exectured prior to training.

Wraps up most of the functions of this class: load the weights if any are given, generaters names for session and weights, creates directories and prints the training session.

