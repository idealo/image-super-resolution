## class DataHandler
DataHandler generate agumented batches used for training or validation.


##### Args
* **lr_dir**: directory containing the Low Res images.

* **hr_dir**: directory containing the High Res images.

* **patch_size**: integer, size of the patches extracted from LR images.

* **scale**: integer, upscaling factor.

* **n_validation_samples**: integer, size of the validation set. Only provided if the
    DataHandler is used to generate validation sets.

* **T**: float in [0,1], is the patch "flatness" threshold.
    Determines what level of detail the patches need to meet. 0 means any patch is accepted.

### \_\_init\_\_
```python
def __init__(lr_dir, hr_dir, patch_size, scale, n_validation_samples, T)
```

### get\_batch
```python
def get_batch(batch_size, idx)
```
Returns a dictionary with keys ('lr', 'hr') containing training batches of Low Res and Highr Res image patches.



### get\_validation\_batches
```python
def get_validation_batches(batch_size)
```
Returns a batch for each image in the validation set.



### get\_validation\_set
```python
def get_validation_set(batch_size)
```
Returns a batch for each image in the validation set. Flattens and splits them to feed it to Keras's model.evaluate.



