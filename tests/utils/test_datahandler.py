import os
import unittest
import numpy as np
from unittest.mock import patch
from ISR.utils.datahandler import DataHandler


class DataHandlerTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def fake_folders(self, kind):
        if kind['matching'] == False:
            if kind['res'] == 'hr':
                return ['data2.gif', 'data1.png', 'data0.jpeg']
            elif kind['res'] == 'lr':
                return ['data1.png']
            else:
                raise
        if kind['matching'] == True:
            if kind['res'] == 'hr':
                return ['data2.gif', 'data1.png', 'data0.jpeg']
            elif kind['res'] == 'lr':
                return ['data1.png', 'data0.jpeg']
            else:
                raise

    def path_giver(self, d, b):
        if d['res'] == 'hr':
            return 'hr'
        else:
            return 'lr'

    def image_getter(self, res):
        if res == 'hr':
            return np.random.random((20, 20, 3))
        else:
            return np.random.random((10, 10, 3))

    def test__make_img_list_non_validation(self):
        with patch('os.listdir', side_effect=self.fake_folders):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                DH = DataHandler(
                    lr_dir={'res': 'lr', 'matching': False},
                    hr_dir={'res': 'hr', 'matching': False},
                    patch_size=0,
                    scale=0,
                    n_validation_samples=None,
                )

        expected_ls = {'hr': ['data0.jpeg', 'data1.png'], 'lr': ['data1.png']}
        self.assertTrue(np.all(DH.img_list['hr'] == expected_ls['hr']))
        self.assertTrue(np.all(DH.img_list['lr'] == expected_ls['lr']))

    def test__make_img_list_validation(self):
        with patch('os.listdir', side_effect=self.fake_folders):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                with patch('numpy.random.choice', return_value=np.array([0])):
                    DH = DataHandler(
                        lr_dir={'res': 'lr', 'matching': False},
                        hr_dir={'res': 'hr', 'matching': False},
                        patch_size=0,
                        scale=0,
                        n_validation_samples=10,
                    )

        expected_ls = {'hr': ['data0.jpeg'], 'lr': ['data1.png']}
        self.assertTrue(np.all(DH.img_list['hr'] == expected_ls['hr']))
        self.assertTrue(np.all(DH.img_list['lr'] == expected_ls['lr']))

    def test__check_dataset_with_mismatching_data(self):
        try:
            with patch('os.listdir', side_effect=self.fake_folders):

                DH = DataHandler(
                    lr_dir={'res': 'lr', 'matching': False},
                    hr_dir={'res': 'hr', 'matching': False},
                    patch_size=0,
                    scale=0,
                    n_validation_samples=None,
                )
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test__check_dataset_with_matching_data(self):
        with patch('os.listdir', side_effect=self.fake_folders):
            DH = DataHandler(
                lr_dir={'res': 'lr', 'matching': True},
                hr_dir={'res': 'hr', 'matching': True},
                patch_size=0,
                scale=0,
                n_validation_samples=None,
            )

    def test__not_flat_with_flat_patch(self):
        lr_patch = np.zeros((5, 5, 3))
        with patch('ISR.utils.datahandler.DataHandler._make_img_list', return_value=True):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                DH = DataHandler(
                    lr_dir=None, hr_dir=None, patch_size=0, scale=0, n_validation_samples=None
                )
        self.assertFalse(DH._not_flat(lr_patch, flatness=0.1))

    def test__not_flat_with_non_flat_patch(self):
        lr_patch = np.random.random((5, 5, 3))
        with patch('ISR.utils.datahandler.DataHandler._make_img_list', return_value=True):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                DH = DataHandler(
                    lr_dir=None, hr_dir=None, patch_size=0, scale=0, n_validation_samples=None
                )
        self.assertTrue(DH._not_flat(lr_patch, flatness=0.00001))

    def test__crop_imgs_crops_shapes(self):
        with patch('ISR.utils.datahandler.DataHandler._make_img_list', return_value=True):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                DH = DataHandler(
                    lr_dir=None, hr_dir=None, patch_size=3, scale=2, n_validation_samples=None
                )
        imgs = {'hr': np.random.random((20, 20, 3)), 'lr': np.random.random((10, 10, 3))}
        crops = DH._crop_imgs(imgs, batch_size=2, flatness=0)
        self.assertTrue(crops['hr'].shape == (2, 6, 6, 3))
        self.assertTrue(crops['lr'].shape == (2, 3, 3, 3))

    def test__apply_transorm(self):
        I = np.ones((2, 2))
        A = I * 0
        B = I * 1
        C = I * 2
        D = I * 3
        image = np.block([[A, B], [C, D]])
        with patch('ISR.utils.datahandler.DataHandler._make_img_list', return_value=True):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                DH = DataHandler(
                    lr_dir=None, hr_dir=None, patch_size=3, scale=2, n_validation_samples=None
                )
        transf = [[1, 0], [0, 1], [2, 0], [0, 2], [1, 1], [0, 0]]
        self.assertTrue(np.all(np.block([[C, A], [D, B]]) == DH._apply_transform(image, transf[0])))
        self.assertTrue(np.all(np.block([[C, D], [A, B]]) == DH._apply_transform(image, transf[1])))
        self.assertTrue(np.all(np.block([[B, D], [A, C]]) == DH._apply_transform(image, transf[2])))
        self.assertTrue(np.all(np.block([[B, A], [D, C]]) == DH._apply_transform(image, transf[3])))
        self.assertTrue(np.all(np.block([[D, B], [C, A]]) == DH._apply_transform(image, transf[4])))
        self.assertTrue(np.all(image == DH._apply_transform(image, transf[5])))

    def test__transform_batch(self):
        with patch('ISR.utils.datahandler.DataHandler._make_img_list', return_value=True):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                DH = DataHandler(
                    lr_dir=None, hr_dir=None, patch_size=3, scale=2, n_validation_samples=None
                )
        I = np.ones((2, 2))
        A = I * 0
        B = I * 1
        C = I * 2
        D = I * 3
        image = np.block([[A, B], [C, D]])
        t_image_1 = np.block([[D, B], [C, A]])
        t_image_2 = np.block([[B, D], [A, C]])
        batch = np.array([image, image])
        expected = np.array([t_image_1, t_image_2])
        self.assertTrue(np.all(DH._transform_batch(batch, [[1, 1], [2, 0]]) == expected))

    def test_get_batch_shape_and_diversity(self):
        patch_size = 3
        with patch('os.listdir', side_effect=self.fake_folders):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                DH = DataHandler(
                    lr_dir={'res': 'lr', 'matching': True},
                    hr_dir={'res': 'hr', 'matching': True},
                    patch_size=patch_size,
                    scale=2,
                    n_validation_samples=None,
                )

        with patch('imageio.imread', side_effect=self.image_getter):
            with patch('os.path.join', side_effect=self.path_giver):
                batch = DH.get_batch(batch_size=5)

        self.assertTrue(type(batch) is dict)
        self.assertTrue(batch['hr'].shape == (5, patch_size * 2, patch_size * 2, 3))
        self.assertTrue(batch['lr'].shape == (5, patch_size, patch_size, 3))

        self.assertTrue(
            np.any(
                [
                    batch['lr'][0] != batch['lr'][1],
                    batch['lr'][1] != batch['lr'][2],
                    batch['lr'][2] != batch['lr'][3],
                    batch['lr'][3] != batch['lr'][4],
                ]
            )
        )

    def test_get_validation_batches_invalid_number_of_samples(self):
        patch_size = 3
        with patch('os.listdir', side_effect=self.fake_folders):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                DH = DataHandler(
                    lr_dir={'res': 'lr', 'matching': True},
                    hr_dir={'res': 'hr', 'matching': True},
                    patch_size=patch_size,
                    scale=2,
                    n_validation_samples=None,
                )

        with patch('imageio.imread', side_effect=self.image_getter):
            with patch('os.path.join', side_effect=self.path_giver):
                try:
                    with patch('raise', None):
                        batch = DH.get_validation_batches(batch_size=5)
                except:
                    self.assertTrue(True)
                else:
                    self.assertTrue(False)

    def test_get_validation_batches_requesting_more_than_available(self):
        patch_size = 3
        with patch('os.listdir', side_effect=self.fake_folders):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                try:
                    DH = DataHandler(
                        lr_dir={'res': 'lr', 'matching': True},
                        hr_dir={'res': 'hr', 'matching': True},
                        patch_size=patch_size,
                        scale=2,
                        n_validation_samples=10,
                    )
                except:
                    self.assertTrue(True)
                else:
                    self.assertTrue(False)

    def test_get_validation_batches_valid_request(self):
        patch_size = 3
        with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
            with patch('os.listdir', side_effect=self.fake_folders):

                DH = DataHandler(
                    lr_dir={'res': 'lr', 'matching': True},
                    hr_dir={'res': 'hr', 'matching': True},
                    patch_size=patch_size,
                    scale=2,
                    n_validation_samples=2,
                )

        with patch('imageio.imread', side_effect=self.image_getter):
            with patch('os.path.join', side_effect=self.path_giver):
                batch = DH.get_validation_batches(batch_size=12)

        self.assertTrue(len(batch) is 2)
        self.assertTrue(type(batch) is list)
        self.assertTrue(type(batch[0]) is dict)
        self.assertTrue(batch[0]['hr'].shape == (12, patch_size * 2, patch_size * 2, 3))
        self.assertTrue(batch[0]['lr'].shape == (12, patch_size, patch_size, 3))
        self.assertTrue(batch[1]['hr'].shape == (12, patch_size * 2, patch_size * 2, 3))
        self.assertTrue(batch[1]['lr'].shape == (12, patch_size, patch_size, 3))

    def test_validation_set(self):
        patch_size = 3
        with patch('os.listdir', side_effect=self.fake_folders):

            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                DH = DataHandler(
                    lr_dir={'res': 'lr', 'matching': True},
                    hr_dir={'res': 'hr', 'matching': True},
                    patch_size=patch_size,
                    scale=2,
                    n_validation_samples=2,
                )

        with patch('imageio.imread', side_effect=self.image_getter):
            with patch('os.path.join', side_effect=self.path_giver):
                batch = DH.get_validation_set(batch_size=12)

        self.assertTrue(type(batch) is dict)
        self.assertTrue(len(batch) is 2)
        self.assertTrue(batch['hr'].shape == (24, patch_size * 2, patch_size * 2, 3))
        self.assertTrue(batch['lr'].shape == (24, patch_size, patch_size, 3))
