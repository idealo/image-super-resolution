import os
import unittest
import shutil
import yaml
from unittest.mock import patch
from ISR.utils.train_helper import TrainerHelper
from ISR.models.rrdn import RRDN
from ISR.models.discriminator import Discriminator
from ISR.models.cut_vgg19 import Cut_VGG19


class UtilsClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = yaml.load(open(os.path.join('tests', 'data', 'config.yml'), 'r'))
        cls.RRDN = RRDN(arch_params=cls.setup['rrdn'], patch_size=cls.setup['patch_size'])
        cls.f_ext = Cut_VGG19(patch_size=cls.setup['patch_size'], layers_to_extract=[1, 2])
        cls.discr = Discriminator(patch_size=cls.setup['patch_size'])
        cls.weights_path = {
            'generator': os.path.join(cls.setup['weights_dir'], 'test_gen_weights.hdf5'),
            'discriminator': os.path.join(cls.setup['weights_dir'], 'test_dis_weights.hdf5'),
        }
        cls.TH = TrainerHelper(
            generator=cls.RRDN,
            weights_dir=cls.setup['weights_dir'],
            logs_dir=cls.setup['log_dir'],
            lr_train_dir=cls.setup['lr_input'],
            feature_extractor=cls.f_ext,
            discriminator=cls.discr,
            dataname='TEST',
            pretrained_weights_path={},
            fallback_save_every_n_epochs=2,
        )

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        if os.path.exists('./tests/temporary_test_data'):
            shutil.rmtree('./tests/temporary_test_data')
        if os.path.exists('./log_file'):
            os.remove('./log_file')
        pass

    def test_make_generator_name(self):
        generator_name = self.TH.generator.name + '-C2-D3-G20-G020-T2-x2'
        generated_name = self.TH._make_generator_name()
        assert generator_name == generated_name, 'Generated name: {}, expected: {}'.format(
            generated_name, generator_name
        )

    def test_basename_without_pretrained_weights(self):
        self.TH.generator_description = self.TH._make_generator_name()
        basename = self.TH.generator_description + '_TEST-vgg19-1-2-srgan-large'
        made_basename = self.TH._make_basename()
        assert basename == made_basename, 'Generated name: {}, expected: {}'.format(
            made_basename, basename
        )

    def test_basename_with_pretrained_weights(self):
        basename = 'test_gen_weights_TEST-vgg19-1-2-srgan-large'
        self.TH.pretrained_weights_path = self.weights_path
        made_basename = self.TH._make_basename()
        self.TH.pretrained_weights_path = {}
        assert basename == made_basename, 'Generated name: {}, expected: {}'.format(
            made_basename, basename
        )

    def test_get_basepath(self):
        basepath = 'rrdn-C2-D3-G20-G020-T2-x2/TEST-vgg19-1-2-srgan-large'
        self.TH.generator_description = self.TH._make_generator_name()
        self.TH.basename = self.TH._make_basename()
        made_basepath = self.TH._get_basepath()
        assert basepath == made_basepath, 'Generated path: {}, expected: {}'.format(
            made_basepath, basepath
        )

    def test_mock_dir_creation(self):
        with patch('ISR.utils.train_helper.TrainerHelper._create_dirs', return_value=True):
            self.assertTrue(self.TH._create_dirs())

    def test_mock_callback_paths_creation(self):
        with patch('ISR.utils.train_helper.TrainerHelper._make_callback_paths', return_value=True):
            self.assertTrue(self.TH._make_callback_paths())

    def test_weights_naming(self):
        w_names = {
            'generator': 'rrdn-C2-D3-G20-G020-T2-x2_TEST-vgg19-1-2-srgan-large-e{epoch:03d}.hdf5',
            'discriminator': 'discr-rrdn-C2-D3-G20-G020-T2-x2_TEST-vgg19-1-2-srgan-large-e{epoch:03d}.hdf5',
        }
        self.TH.generator_description = self.TH._make_generator_name()
        self.TH.basename = self.TH._make_basename()
        generated_names = self.TH._weights_name()
        assert (
            w_names['generator'] == generated_names['generator']
        ), 'Generated names: {}, expected: {}'.format(
            generated_names['generator'], w_names['generator']
        )
        assert (
            w_names['discriminator'] == generated_names['discriminator']
        ), 'Generated names: {}, expected: {}'.format(
            generated_names['discriminator'], w_names['discriminator']
        )

    def test_mock_training_setting_printer(self):
        with patch(
            'ISR.utils.train_helper.TrainerHelper.print_training_setting', return_value=True
        ):
            self.assertTrue(self.TH.print_training_setting())

    def test_weights_saving(self):
        w_names = {
            'generator': 'test_gen_weights_TEST-vgg19-1-2-srgan-large-e{epoch:03d}.hdf5',
            'discriminator': 'test_discr_weights_TEST-vgg19-1-2-srgan-large-e{epoch:03d}.hdf5',
        }
        self.TH.generator_description = self.TH._make_generator_name()
        self.TH.basename = self.TH._make_basename()
        self.TH.basepath = self.TH._get_basepath()
        self.TH.weights_name = self.TH._weights_name()
        self.TH.callback_paths = self.TH._make_callback_paths()
        self.TH._create_dirs()
        self.TH._save_weights(1, self.TH.generator.model, self.TH.discriminator, best=False)

        assert os.path.exists(
            './tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/TEST-vgg19-1-2-srgan-large/discr-rrdn-C2-D3-G20-G020-T2-x2_TEST-vgg19-1-2-srgan-large-e002.hdf5'
        )
        assert os.path.exists(
            './tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/TEST-vgg19-1-2-srgan-large/rrdn-C2-D3-G20-G020-T2-x2_TEST-vgg19-1-2-srgan-large-e002.hdf5'
        )

    def test_mock_epoch_end(self):
        with patch('ISR.utils.train_helper.TrainerHelper.on_epoch_end', return_value=True):
            self.assertTrue(self.TH.on_epoch_end())

    def test_epoch_number_from_weights_names(self):
        w_names = {
            'generator': 'test_gen_weights_TEST-vgg19-1-2-srgan-large-e003.hdf5',
            'discriminator': 'test_discr_weights_TEST-vgg19-1-2-srgan-large-e003.hdf5',
        }
        e_n = self.TH.epoch_n_from_weights_name(w_names['generator'])
        assert e_n == 3

    def test_mock_initalize_training(self):
        with patch('ISR.utils.train_helper.TrainerHelper.initialize_training', return_value=True):
            self.assertTrue(self.TH.initialize_training())
