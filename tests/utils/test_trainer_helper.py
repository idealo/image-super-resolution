import unittest
import shutil
import yaml
from pathlib import Path
from unittest.mock import patch
from ISR.utils.train_helper import TrainerHelper
from ISR.models.rrdn import RRDN
from ISR.models.discriminator import Discriminator
from ISR.models.cut_vgg19 import Cut_VGG19


class UtilsClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = yaml.load(Path('./tests/data/config.yml').read_text())
        cls.RRDN = RRDN(arch_params=cls.setup['rrdn'], patch_size=cls.setup['patch_size'])
        cls.f_ext = Cut_VGG19(patch_size=cls.setup['patch_size'], layers_to_extract=[1, 2])
        cls.discr = Discriminator(patch_size=cls.setup['patch_size'])
        cls.weights_path = {
            'generator': Path(cls.setup['weights_dir']) / 'test_gen_weights.hdf5',
            'discriminator': Path(cls.setup['weights_dir']) / 'test_dis_weights.hdf5',
        }
        cls.TH = TrainerHelper(
            generator=cls.RRDN,
            weights_dir=cls.setup['weights_dir'],
            logs_dir=cls.setup['log_dir'],
            lr_train_dir=cls.setup['lr_input'],
            feature_extractor=cls.f_ext,
            discriminator=cls.discr,
            dataname='TEST',
            weights_generator='',
            weights_discriminator='',
            fallback_save_every_n_epochs=2,
        )
        cls.TH.session_id = '0000'
        cls.TH.logger.setLevel(50)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        if Path('./tests/temporary_test_data').exists():
            shutil.rmtree('./tests/temporary_test_data')
        if Path('./log_file').exists():
            Path('./log_file').unlink()
        pass

    def test__make_basename(self):
        generator_name = self.TH.generator.name + '-C2-D3-G20-G020-T2-x2'
        generated_name = self.TH._make_basename()
        assert generator_name == generated_name, 'Generated name: {}, expected: {}'.format(
            generated_name, generator_name
        )

    def test_basename_without_pretrained_weights(self):
        basename = 'rrdn-C2-D3-G20-G020-T2-x2'
        made_basename = self.TH._make_basename()
        assert basename == made_basename, 'Generated name: {}, expected: {}'.format(
            made_basename, basename
        )

    def test_basename_with_pretrained_weights(self):
        basename = 'rrdn-C2-D3-G20-G020-T2-x2'
        self.TH.pretrained_weights_path = self.weights_path
        made_basename = self.TH._make_basename()
        self.TH.pretrained_weights_path = {}
        assert basename == made_basename, 'Generated name: {}, expected: {}'.format(
            made_basename, basename
        )

    def test_callback_paths_creation(self):
        # reset session_id
        self.TH.callback_paths = self.TH._make_callback_paths()
        self.assertTrue(
            self.TH.callback_paths['weights']
            == Path('tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000')
        )
        self.assertTrue(
            self.TH.callback_paths['logs']
            == Path('tests/temporary_test_data/logs/rrdn-C2-D3-G20-G020-T2-x2/0000')
        )

    def test_weights_naming(self):
        w_names = {
            'generator': Path(
                'tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/rrdn-C2-D3-G20-G020-T2-x2{metric}_epoch{epoch:03d}.hdf5'
            ),
            'discriminator': Path(
                'tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/srgan-large{metric}_epoch{epoch:03d}.hdf5'
            ),
        }
        cb_paths = self.TH._make_callback_paths()
        generated_names = self.TH._weights_name(cb_paths)
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

        self.TH.callback_paths = self.TH._make_callback_paths()
        self.TH.weights_name = self.TH._weights_name(self.TH.callback_paths)
        Path('tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/').mkdir(
            parents=True
        )
        self.TH._save_weights(1, self.TH.generator.model, self.TH.discriminator, best=False)

        assert Path(
            './tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/rrdn-C2-D3-G20-G020-T2-x2_epoch002.hdf5'
        ).exists()
        assert Path(
            './tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/srgan-large_epoch002.hdf5'
        ).exists()

    def test_mock_epoch_end(self):
        with patch('ISR.utils.train_helper.TrainerHelper.on_epoch_end', return_value=True):
            self.assertTrue(self.TH.on_epoch_end())

    def test_epoch_number_from_weights_names(self):
        w_names = {
            'generator': 'test_gen_weights_TEST-vgg19-1-2-srgan-large-e003.hdf5',
            'discriminator': 'txxxxxxxxepoch003xxxxxhdf5',
            'discriminator2': 'test_discr_weights_TEST-vgg19-1-2-srgan-large-epoch03.hdf5',
        }
        e_n = self.TH.epoch_n_from_weights_name(w_names['generator'])
        assert e_n == 0
        e_n = self.TH.epoch_n_from_weights_name(w_names['discriminator'])
        assert e_n == 3
        e_n = self.TH.epoch_n_from_weights_name(w_names['discriminator2'])
        assert e_n == 0

    def test_mock_initalize_training(self):
        with patch('ISR.utils.train_helper.TrainerHelper.initialize_training', return_value=True):
            self.assertTrue(self.TH.initialize_training())
