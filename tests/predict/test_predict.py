import logging
import unittest
import yaml
import shutil
import numpy as np
from copy import copy
from pathlib import Path
from ISR.models.rdn import RDN
from ISR.predict.predictor import Predictor
from unittest.mock import patch, Mock


class PredictorClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls.setup = yaml.load(Path('tests/data/config.yml').read_text(), Loader=yaml.FullLoader)
        cls.RDN = RDN(arch_params=cls.setup['rdn'], patch_size=cls.setup['patch_size'])

        cls.temp_data = Path('tests/temporary_test_data')
        cls.valid_files = cls.temp_data / 'valid_files'
        cls.valid_files.mkdir(parents=True, exist_ok=True)
        for item in ['data2.gif', 'data1.png', 'data0.jpeg']:
            (cls.valid_files / item).touch()

        cls.invalid_files = cls.temp_data / 'invalid_files'
        cls.invalid_files.mkdir(parents=True, exist_ok=True)
        for item in ['data2.gif', 'data.data', 'data02']:
            (cls.invalid_files / item).touch()

        def nullifier(*args):
            pass

        cls.out_dir = cls.temp_data / 'out_dir'
        cls.predictor = Predictor(input_dir=str(cls.valid_files), output_dir=str(cls.out_dir))
        cls.predictor.logger = Mock(return_value=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_data)
        pass

    def setUp(self):
        self.pred = copy(self.predictor)
        pass

    def tearDown(self):
        pass

    def test__load_weights_with_no_weights(self):
        self.pred.weights_path = None
        try:
            self.pred._load_weights()
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test__load_weights_with_valid_weights(self):
        def raise_path(path):
            raise ValueError(path)

        self.pred.model = self.RDN
        self.pred.model.model.load_weights = Mock(side_effect=raise_path)
        self.pred.weights_path = 'a/path'
        try:
            self.pred._load_weights()
        except ValueError as e:
            self.assertTrue(str(e) == 'a/path')
        else:
            self.assertTrue(False)

    def test__make_basename(self):
        self.pred.model = self.RDN
        made_name = self.pred._make_basename()
        self.assertTrue(made_name == 'rdn-C3-D10-G64-G064-x2')

    def test__forward_pass_pixel_range_and_type(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr

        self.pred.model = self.RDN
        self.pred.model.model.predict = Mock(side_effect=valid_sr_output)
        with patch('imageio.imread', return_value=np.random.random((10, 10, 3))):
            sr = self.pred._forward_pass('file_path')
        self.assertTrue(type(sr[0, 0, 0]) is np.uint8)
        self.assertTrue(np.all(sr >= 0.0))
        self.assertTrue(np.all(sr <= 255.0))
        self.assertTrue(np.any(sr > 1.0))
        self.assertTrue(sr.shape == (20, 20, 3))

    def test__forward_pass_4_channela(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr

        self.pred.model = self.RDN
        self.pred.model.model.predict = Mock(side_effect=valid_sr_output)
        with patch('imageio.imread', return_value=np.random.random((10, 10, 4))):
            sr = self.pred._forward_pass('file_path')
        self.assertTrue(sr is None)

    def test__forward_pass_1_channel(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr

        self.pred.model = self.RDN
        self.pred.model.model.predict = Mock(side_effect=valid_sr_output)
        with patch('imageio.imread', return_value=np.random.random((10, 10, 1))):
            sr = self.pred._forward_pass('file_path')
        self.assertTrue(sr is None)

    def test_get_predictions(self):
        self.pred._load_weights = Mock(return_value={})
        self.pred._forward_pass = Mock(return_value=True)
        with patch('imageio.imwrite', return_value=True):
            self.pred.get_predictions(self.RDN, 'a/path/arch-weights_session1_session2.hdf5')
        pass

    def test_output_folder_and_dataname(self):
        self.assertTrue(self.pred.data_name == 'valid_files')
        self.assertTrue(
            self.pred.output_dir == Path('tests/temporary_test_data/out_dir/valid_files')
        )

    def test_valid_extensions(self):
        self.assertTrue(
            np.array_equal(
                np.sort(self.pred.img_ls),
                np.sort([self.valid_files / 'data0.jpeg', self.valid_files / 'data1.png']),
            )
        )

    def test_no_valid_images(self):
        try:
            predictor = Predictor(input_dir=str(self.invalid_files), output_dir=str(self.out_dir))
        except ValueError as e:
            self.assertTrue('image' in str(e))
        else:
            self.assertTrue(False)
