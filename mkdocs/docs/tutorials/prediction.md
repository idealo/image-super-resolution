# ISR Suite: HOW-TO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idealo/image-super-resolution/blob/master/notebooks/ISR_Prediction_Tutorial.ipynb)

## Prediction

### Get the pre-trained weights and data
Get the weights with
```bash
wget https://github.com/idealo/image-super-resolution/raw/master/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5
wget https://github.com/idealo/image-super-resolution/raw/master/weights/sample_weights/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5
wget https://github.com/idealo/image-super-resolution/raw/master/weights/sample_weights/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5
mkdir weights
mv *.hdf5 weights
```
Download a sample image, in this case
```bash
wget http://images.math.cnrs.fr/IMG/png/section8-image.png
mkdir -p data/input/test_images
mv *.png data/input/test_images
```

Load the image with PIL, scale it and convert it into a format our model can use (it needs the extra dimension)
```python
import numpy as np
from PIL import Image

img = Image.open('data/input/test_images/section8-image.png')
lr_img = np.array(img)
```

### Get predictions

#### Create the model and run prediction
Create the RDN model, for which we provide pre-trained weights, and load them.<br>
Choose amongst the available model weights, compare the output if you wish.

```python
from ISR.models import RDN
```
##### Large RDN model

```python
rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5')
```

##### Small RDN model

```python
rdn = RDN(arch_params={'C':3, 'D':10, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5')
```

##### Large RDN noise cancelling, detail enhancing model

```python
rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
```

##### Run prediction

```python
sr_img = rdn.predict(lr_img)
Image.fromarray(sr_img)
```

#### Usecase: upscaling noisy images

Now, for science, let's make it harder for the networks.

We compress the image into the jpeg format to introduce compression artefact and lose some information.

We will compare:

- the baseline bicubic scaling
- the basic model - Add Hyperlink
- a model trained to remove noise using perceptual loss with deep features and GANs training

So let's first  compress the image


```python
img.save('data/input/test_images/compressed.jpeg','JPEG', dpi=[300, 300], quality=50)
compressed_img = Image.open('data/input/test_images/compressed.jpeg')
compressed_lr_img = np.array(compressed_img)
compressed_img.show()
```

##### Baseline
Bicubic scaling
```python
compressed_img.resize(size=(compressed_img.size[0]*2, compressed_img.size[1]*2), resample=Image.BICUBIC)
```

##### Large RDN model (PSNR trained)

```python
rdn = RDN(arch_params={'C': 6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5')
sr_img = rdn.predict(compressed_lr_img)
Image.fromarray(sr_img)
```

##### Small RDN model  (PSNR trained)

```python
rdn = RDN(arch_params={'C': 3, 'D':10, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5')
sr_img = rdn.predict(compressed_lr_img)
Image.fromarray(sr_img)
```

##### Large RDN noise cancelling, detail enhancing model

```python
rdn = RDN(arch_params={'C': 6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
sr_img = rdn.predict(compressed_lr_img)
Image.fromarray(sr_img)
```

#### Predictor Class
You can also use the predictor class to run the model on entire folders.  To do so you first need to create an output folder to collect your results, in this case `data/output`:

```python
from ISR.predict import Predictor
predictor = Predictor(input_dir='data/input/test_images/', output_dir='data/output')
predictor.get_predictions(model=rdn, weights_path='weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
```
