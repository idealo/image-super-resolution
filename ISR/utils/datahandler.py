import os
import imageio
import numpy as np
from ISR.utils.logger import get_logger


class DataHandler:
    """
    DataHandler generate augmented batches used for training or validation.

    Args:
        lr_dir: directory containing the Low Res images.
        hr_dir: directory containing the High Res images.
        patch_size: integer, size of the patches extracted from LR images.
        scale: integer, upscaling factor.
        n_validation_samples: integer, size of the validation set. Only provided if the
            DataHandler is used to generate validation sets.
        T: float in [0,1], is the patch "flatness" threshold.
            Determines what level of detail the patches need to meet. 0 means any patch is accepted.
    """

    def __init__(self, lr_dir, hr_dir, patch_size, scale, n_validation_samples=None, T=0.03):
        self.folders = {'hr': hr_dir, 'lr': lr_dir}  # image folders
        self.extensions = ('.png', '.jpeg', '.jpg')  # admissible extension
        self.img_list = {}  # list of file names
        self.n_validation_samples = n_validation_samples
        self.patch_size = patch_size
        self.scale = scale
        self.T = T
        self.patch_size = {'lr': patch_size, 'hr': patch_size * self.scale}
        self.logger = get_logger(__name__)
        self._make_img_list()
        self._check_dataset()

    def _make_img_list(self):
        """ Creates a dictionary of lists of the acceptable images contained in lr_dir and hr_dir. """

        for res in ['hr', 'lr']:
            file_names = os.listdir(self.folders[res])
            file_names = [file for file in file_names if file.endswith(self.extensions)]
            self.img_list[res] = np.sort(file_names)

        if self.n_validation_samples:
            samples = np.random.choice(
                range(len(self.img_list['hr'])), self.n_validation_samples, replace=False
            )
            for res in ['hr', 'lr']:
                self.img_list[res] = self.img_list[res][samples]

    def _check_dataset(self):
        """ Sanity check for dataset. """

        # the order of these asserts is important for testing
        assert len(self.img_list['hr']) == self.img_list['hr'].shape[0], 'UnevenDatasets'
        assert self._matching_datasets(), 'Input/LabelsMismatch'

    def _matching_datasets(self):
        """ Rough file name matching between lr and hr directories. """
        # LR_name.png = HR_name+x+scale.png
        # or
        # LR_name.png = HR_name.png
        LR_name_root = [x.split('.')[0].split('x')[0] for x in self.img_list['lr']]
        HR_name_root = [x.split('.')[0] for x in self.img_list['hr']]
        return np.all(HR_name_root == LR_name_root)

    def _not_flat(self, patch):
        """
        Determines whether the patch is complex, or not-flat enough.
        Threshold set by self.T.
        """

        if max(np.std(patch, axis=0).mean(), np.std(patch, axis=1).mean()) < self.T:
            return False
        else:
            return True

    def _crop_imgs(self, imgs, batch_size, idx=0):
        """
        Get random top left corners coordinates in LR space, multiply by scale to
        get HR coordinates.
        Gets batch_size + n possible coordinates.
        Accepts the batch only if the standard deviation of pixel intensities is above a given threshold, OR
        no patches can be further discarded (n have been discarded already).
        Square crops of size patch_size are taken from the selected
        top left corners.
        """

        n = 2 * batch_size
        top_left = {'x': {}, 'y': {}}
        for i, axis in enumerate(['x', 'y']):
            top_left[axis]['lr'] = np.random.randint(
                0, imgs['lr'].shape[i] - self.patch_size['lr'] + 1, batch_size + n
            )
            top_left[axis]['hr'] = top_left[axis]['lr'] * self.scale

        crops = {}
        for res in ['lr', 'hr']:
            slices = [
                [slice(x, x + self.patch_size[res]), slice(y, y + self.patch_size[res])]
                for x, y in zip(top_left['x'][res], top_left['y'][res])
            ]
            crops[res] = []
            for s in slices:
                candidate_crop = imgs[res][s[0], s[1], slice(None)]
                if self._not_flat(candidate_crop) or n == 0:
                    crops[res].append(candidate_crop)
                else:
                    n -= 1
                if len(crops[res]) == batch_size:
                    break
            crops[res] = np.array(crops[res])
        return crops

    def _apply_transform(self, img, transform_selection):
        """ Rotates and flips input image according to transform_selection. """

        rotate = {
            0: lambda x: x,
            1: lambda x: np.rot90(x, k=1, axes=(1, 0)),  # rotate right
            2: lambda x: np.rot90(x, k=1, axes=(0, 1)),  # rotate left
        }

        flip = {
            0: lambda x: x,
            1: lambda x: np.flip(x, 0),  # flip along horizontal axis
            2: lambda x: np.flip(x, 1),  # flip along vertical axis
        }

        rot_direction = transform_selection[0]
        flip_axis = transform_selection[1]

        img = rotate[rot_direction](img)
        img = flip[flip_axis](img)

        return img

    def _transform_batch(self, batch, transforms):
        """ Transforms each individual image of the batch independently. """

        t_batch = np.array(
            [self._apply_transform(img, transforms[i]) for i, img in enumerate(batch)]
        )
        return t_batch

    def get_batch(self, batch_size, idx=None):
        """
        Returns a dictionary with keys ('lr', 'hr') containing training batches
        of Low Res and High Res image patches.
        """

        if not idx:
            # randomly select one image. idx is given at validation time.
            idx = np.random.choice(range(len(self.img_list['hr'])))
        img = {}
        for res in ['lr', 'hr']:
            img_path = os.path.join(self.folders[res], self.img_list[res][idx])
            img[res] = imageio.imread(img_path) / 255.0
        batch = self._crop_imgs(img, batch_size)
        transforms = np.random.randint(0, 3, (batch_size, 2))
        batch['lr'] = self._transform_batch(batch['lr'], transforms)
        batch['hr'] = self._transform_batch(batch['hr'], transforms)

        return batch

    def get_validation_batches(self, batch_size):
        """ Returns a batch for each image in the validation set. """

        if self.n_validation_samples:
            batches = []
            for idx in range(self.n_validation_samples):
                batches.append(self.get_batch(batch_size, idx))
            return batches
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )

    def get_validation_set(self, batch_size):
        """
        Returns a batch for each image in the validation set.
        Flattens and splits them to feed it to Keras's model.evaluate.
        """

        if self.n_validation_samples:
            batches = self.get_validation_batches(batch_size)
            valid_set = {'lr': [], 'hr': []}
            for batch in batches:
                for res in ('lr', 'hr'):
                    valid_set[res].extend(batch[res])
            for res in ('lr', 'hr'):
                valid_set[res] = np.array(valid_set[res])
            return valid_set
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )
