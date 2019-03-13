## class Cut_VGG19
Class object that fetches the pre trained VGG19 model and declares <layers_to_extract> as output layers.


##### Args
* **layers_to_extract**: list of layers to be declared as output layers.

* **patch_size**: integer, defines the size of the input (patch_size x patch_size).

##### Attributes
* **loss_model**: multi-output vgg architecure with <layers_to_extract> as output layers.

### \_\_init\_\_
```python
def __init__(patch_size, layers_to_extract)
```

