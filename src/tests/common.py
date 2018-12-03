import os
import shutil
import imageio
import unittest
import numpy as np


class TestWithData(unittest.TestCase):
    def dataSetUp(self):
        self.img_size = {'HR': 64, 'LR': 32}
        self.patch_size = {'HR': 32, 'LR': 16}
        self.scale = 2
        self.dataset_folder = {}
        self.dataset_types = ['correct', 'uneven', 'mismatch']
        self.temp_folder = 'tests/temporary_test_data'
        if not os.path.exists(self.temp_folder):
            os.mkdir(self.temp_folder)
        self.results_folder = os.path.join(self.temp_folder, 'results')
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)
        self.weights_folder = os.path.join(self.temp_folder, 'test_weights')
        if not os.path.exists(self.weights_folder):
            os.mkdir(self.weights_folder)
        self.dataset_path = os.path.join(self.temp_folder, '{}_x{}')
        pass

    def dataTearDown(self):
        shutil.rmtree(self.temp_folder)
        pass

    def create_random_dataset(self, type, batch_size=8, dataset_size=16):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.dataset_folder[type] = {
            'LR': self.dataset_path.format(type, str(self.scale)),
            'HR': self.dataset_path.format(type, '1'),
        }

        file_path = {}
        for res in ['LR', 'HR']:
            if not os.path.exists(self.dataset_folder[type][res]):
                os.mkdir(self.dataset_folder[type][res])
            file_path[res] = os.path.join(self.dataset_folder[type][res], 'img{}.png')

        for i in range(self.dataset_size):
            img = np.random.randint(0, 256, (self.img_size['HR'], self.img_size['HR'], 3)).astype(np.uint8)
            scaled_img = img[:: self.scale, :: self.scale, :]

            if (type == 'uneven') and (i == 0):
                imageio.imwrite(file_path['HR'].format(str(i)), img)
            elif (type == 'mismatch') and (i == 0):
                imageio.imwrite(file_path['HR'].format(str(i)), img)
                imageio.imwrite(file_path['LR'].format(str(self.dataset_size)), scaled_img)
            else:
                imageio.imwrite(file_path['HR'].format(str(i)), img)
                imageio.imwrite(file_path['LR'].format(str(i)), scaled_img)
