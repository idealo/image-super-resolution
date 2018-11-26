import os
import imageio
import numpy as np
import keras.backend as K
from utils.metrics import PSNR
from utils.generator import Generator
from tests.common import TestWithData


class UtilsClassTest(TestWithData):
    def setUp(self):
        self.dataSetUp()
        pass

    def tearDown(self):
        self.dataTearDown()

    def get_generator(self, generator_type, dataset_type):
        """Creates a generator of the specified type, with the specified
        dataset type
        """
        self.create_random_dataset(dataset_type)

        generator = Generator(input_folder=self.dataset_folder[dataset_type]['LR'],
                              label_folder=self.dataset_folder[dataset_type]['HR'],
                              patch_size=self.patch_size['LR'],
                              batch_size=self.batch_size,
                              mode=generator_type,
                              scale=self.scale)

        return generator

    def test_rotation(self):
        generator = self.get_generator('train', 'correct')
        test_image_path = os.path.join(generator.folder['LR'],
                                       generator.img_list['LR'][0])
        test_image = imageio.imread(test_image_path)

        rotations = {0: lambda x: x,  # no rotation
                     1: lambda x: np.rot90(x, k=1, axes=(1, 0)),  # rotate right
                     2: lambda x: np.rot90(x, k=1, axes=(0, 1))}  # rotate left

        for rot in rotations.keys():
            transf_check = rotations[rot](test_image)
            gen_transf = generator._apply_transform(test_image, [rot, 0])
            self.assertTrue(np.all(gen_transf == transf_check))

    def test_flip(self):
        generator = self.get_generator('train', 'correct')
        test_image_path = os.path.join(generator.folder['LR'],
                                       generator.img_list['LR'][0])
        test_image = imageio.imread(test_image_path)

        flips = {0: lambda x: x,  # no flips
                 1: lambda x: np.flip(x, 0),  # flip along horizontal axis
                 2: lambda x: np.flip(x, 1)}  # flip along vertical axis

        for flip in flips.keys():
            transf_check = flips[flip](test_image)
            gen_transf = generator._apply_transform(test_image, [0, flip])
            self.assertTrue(np.all(gen_transf == transf_check))

    def test_train_batch_shapes(self):
        train_generator = self.get_generator('train', 'correct')
        x, y = train_generator.__getitem__(0)
        self.assertTrue(np.all(x.shape == (self.batch_size, self.patch_size['LR'], self.patch_size['LR'], 3)))
        self.assertTrue(np.all(y.shape == (self.batch_size, self.patch_size['HR'], self.patch_size['HR'], 3)))

    def test_validation_batch_shapes(self):
        valid_generator = self.get_generator('valid', 'correct')
        x, y = valid_generator.__getitem__(0)
        self.assertTrue(np.all(x.shape == (self.batch_size, self.patch_size['LR'], self.patch_size['LR'], 3)))
        self.assertTrue(np.all(y.shape == (self.batch_size, self.patch_size['HR'], self.patch_size['HR'], 3)))

    def test_uneven_datasets_detection(self):
        with self.assertRaisesRegex(Exception, 'UnevenDatasets') as cm:
            _ = self.get_generator('train', 'uneven')
        with self.assertRaisesRegex(Exception, 'UnevenDatasets') as cm:
            _ = self.get_generator('valid', 'uneven')

    def test_that_a_correct_dataset_does_not_break_generator(self):
        try:
            _ = self.get_generator('train', 'correct')
        except:
            self.fail('Expection raised for loading a correct dataset')
        try:
            _ = self.get_generator('valid', 'correct')
        except:
            self.fail('Expection raised for loading a correct dataset')

    def test_mismatching_datasets_detection(self):
        with self.assertRaisesRegex(Exception, 'Input/LabelsMismatch') as cm:
            _ = self.get_generator('train', 'mismatch')

        with self.assertRaisesRegex(Exception, 'Input/LabelsMismatch') as cm:
            _ = self.get_generator('valid', 'mismatch')

    def test_training_batch_pixel_type_and_range(self):
        train_generator = self.get_generator('train', 'correct')
        x, y = train_generator.__getitem__(0)
        for batch in [x, y]:
            self.assertTrue(np.all(batch <= 1.) and np.all(batch >= 0.))
            self.assertTrue(type(batch[0, 0, 0, 0]) == np.float64)

    def test_validation_batch_pixel_type_and_range(self):
        valid_generator = self.get_generator('valid', 'correct')
        x, y = valid_generator.__getitem__(0)
        for batch in [x, y]:
            self.assertTrue(np.all(batch <= 1.) and np.all(batch >= 0.))
            self.assertTrue(type(batch[0, 0, 0, 0]) == np.float64)

    def test_PSNR_sanity(self):
        A = K.ones((10, 10, 3))
        B = K.zeros((10, 10, 3))
        self.assertEqual(K.get_value(PSNR(A, A)), np.inf)
        self.assertEqual(K.get_value(PSNR(A, B)), 0)

    def test_index_shuffle_on_epoch_end(self):
        generator = {}
        idx_before_end = {}
        for type in ['train', 'valid']:
            generator[type] = self.get_generator(type, 'correct')
            idx_before_end[type] = np.copy(generator[type].indices)
            generator[type].on_epoch_end()
            self.assertEqual(len(generator[type].indices), self.dataset_size)
        self.assertFalse(np.all(idx_before_end['train'] == generator['train'].indices))
        self.assertFalse(np.any(idx_before_end['valid'] != generator['valid'].indices))

    def test_deterministic_validation_batches_and_random_train_batches(self):
        generator = {}
        control_batches = {}
        generated_batch = {}
        for type in ['train', 'valid']:
            generator[type] = self.get_generator(type, 'correct')
            control_batches[type] = generator[type].get_batches(0)
            generated_batch[type] = dict(zip(['LR', 'HR'], generator[type].__getitem__(0)))
        self.assertTrue(np.all(control_batches['valid']['LR'] == generated_batch['valid']['LR']))
        self.assertTrue(np.any(control_batches['train']['LR'] != generated_batch['train']['LR']))

        generator['valid'].on_epoch_end()
        generated_batch['valid'] = dict(zip(['LR', 'HR'], generator['valid'].__getitem__(0)))
        self.assertTrue(np.all(control_batches['valid']['LR'] == generated_batch['valid']['LR']))
