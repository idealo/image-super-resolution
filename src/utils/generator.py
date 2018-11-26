import os
import numpy as np
import keras.backend as K
from imageio import imread
from keras.utils import Sequence

class Generator(Sequence):
    """Keras Sequence object to train a model on larger-than-memory data.
    Creates list of LR input and HR label files (images) locations.
    Lists are filtered by extension and sorted by name.

    During training takes:
    >> 1 LR image and the corresponding HR image

    Returns:
    >> 2 batches of batch_size, 1 for each LR and HR
    >> each batch contains batch_size patches of size patch_size
    >>>> extracted from the one LR image
    >> each patch in batch is agumented with a random combination of
    >>>> 1 90degree rotation and 1 horizontal/vertical flip

    self.index is of the size of the original HR dataset
      (g.e.800 images -> self.len = 800)

    the index passed to __getitem__ is of a single image

    """

    def __init__(self, input_folder, label_folder, batch_size, mode='train',
                 patch_size=None, scale=None, n_validation_samples=None):
        self.mode = mode  # shuffle when in train mode
        self.scale = scale  # scaling factor LR to HR
        self.batch_size = batch_size  # batch size
        self.extensions = ['.png', '.jpeg', '.jpg']  # admissible extension
        self.folder = {'HR': label_folder,
                       'LR': input_folder}  # image folders
        # size of patches to extract from LR images
        self.patch_size = {'LR': patch_size,
                           'HR': patch_size * self.scale}

        self.img_list = {}  # list of file names
        for res in ['HR', 'LR']:
            file_names = os.listdir(self.folder[res])
            file_names = [file for file in file_names
                          if any(file.lower().endswith(ext) for ext in self.extensions)]

            if self.mode is 'valid':
                self.img_list[res] = np.sort(file_names)[0:n_validation_samples]
            else:
                self.img_list[res] = np.sort(file_names)

        # order of asserts is important for testing
        assert self.img_list['HR'].shape[0] == self.img_list['LR'].shape[0], \
            'UnevenDatasets'
        assert self.matching_datasets(), 'Input/LabelsMismatch'

        self.indices = np.arange(self.img_list['HR'].shape[0])  # indexes list

    def matching_datasets(self):
        # LR_name.png = HR_name+x+scale.png
        LR_name_root = [x.split('.')[0].split('x')[0] for x in self.img_list['LR']]
        HR_name_root = [x.split('.')[0] for x in self.img_list['HR']]
        return np.all(HR_name_root == LR_name_root)

    def __len__(self):
        # compute number of batches to yield
        return len(self.img_list['HR'])

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        if (self.mode == 'train'):
            np.random.shuffle(self.indices)

    def _crop_imgs(self, imgs, idx=0):
        """Get top left corners coordinates in LR space, multiply by scale to
        get HR coordinates.
        During training the corners are randomly chosen.
        During validation the corners are chosen randomly according to
        a specific seed.
        Square crops of size patch_size are taken from the selected
        top left corners.
        """
        top_left = {'x': {},
                    'y': {}}
        for i, axis in enumerate(['x', 'y']):
            if self.mode == 'train':
                top_left[axis]['LR'] = np.random.randint(0,
                                                         imgs['LR'].shape[i] - self.patch_size['LR'] + 1,
                                                         self.batch_size)
            if self.mode == 'valid':
                not_random = np.random.RandomState(idx)
                top_left[axis]['LR'] = not_random.randint(0,
                                                          imgs['LR'].shape[i] - self.patch_size['LR'] + 1,
                                                          self.batch_size)
            top_left[axis]['HR'] = top_left[axis]['LR'] * self.scale

        crops = {}
        for res in ['LR', 'HR']:
            slices = [[slice(x, x + self.patch_size[res]), slice(y, y + self.patch_size[res])]
                      for x, y in zip(top_left['x'][res], top_left['y'][res])]
            crops[res] = np.array([imgs[res][s[0], s[1], slice(None)] for s in slices])
        return crops

    def get_batches(self, idx):
        """Fetch a batch of images LR and HR.
        Takes #batch_size random patches from 1 LR and 1 HR image,
        returns the patches as a batch.
        """
        imgs = {}
        for res in ['LR', 'HR']:
            imgs[res] = imread(os.path.join(self.folder[res], self.img_list[res][idx])) / 255.

        return self._crop_imgs(imgs, idx)

    def _apply_transform(self, img, transform_selection):
        """Rotates and flips input image according to transform_selection."""
        rotate = {0: lambda x: x,
                  1: lambda x: np.rot90(x, k=1, axes=(1, 0)),  # rotate right
                  2: lambda x: np.rot90(x, k=1, axes=(0, 1))}  # rotate left

        flip = {0: lambda x: x,
                1: lambda x: np.flip(x, 0),  # flip along horizontal axis
                2: lambda x: np.flip(x, 1)}  # flip along vertical axis

        rot_direction = transform_selection[0]
        flip_axis = transform_selection[1]

        img = rotate[rot_direction](img)
        img = flip[flip_axis](img)

        return img

    def _transform_batch(self, batch, transforms):
        """Transforms each individual image of the batch independently."""
        t_batch = np.array([self._apply_transform(img, transforms[i])
                            for i, img in enumerate(batch)])
        return t_batch

    def __getitem__(self, idx):
        # idx is batch index
        idx = self.indices[idx]
        batches = self.get_batches(idx)
        if self.mode == 'train':
            # Select the random transformations to apply
            transforms = np.random.randint(0, 3, (self.batch_size, 2))
            batches['LR'] = self._transform_batch(batches['LR'], transforms)
            batches['HR'] = self._transform_batch(batches['HR'], transforms)

        return batches['LR'], batches['HR']
