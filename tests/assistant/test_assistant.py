import logging
import os
import unittest
import yaml
from ISR import assistant
from unittest.mock import patch, Mock


class Object:
    def __init__(self, *args, **kwargs):
        self.scale = 0
        self.patch_size = 0
        pass

    def make_model(self, *args, **kwargs):
        return self

    def train(self, *args, **kwargs):
        return True

    def get_predictions(self, *args, **kwargs):
        return True


class RunFunctionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        conf = yaml.load(open(os.path.join('tests', 'data', 'config.yml'), 'r'))
        conf['default'] = {
            'feature_extractor': False,
            'discriminator': False,
            'generator': 'rdn',
            'training_set': 'test',
            'test_set': 'test',
        }
        conf['session'] = {}
        conf['session']['training'] = {}
        conf['session']['training']['patch_size'] = 0
        conf['session']['training']['epochs'] = 0
        conf['session']['training']['steps_per_epoch'] = 0
        conf['session']['training']['batch_size'] = 0
        conf['session']['prediction'] = {}
        conf['session']['prediction']['patch_size'] = 5
        conf['generators'] = {}
        conf['generators']['rdn'] = {}
        conf['generators']['rdn']['x'] = 0
        conf['training_sets'] = {}
        conf['training_sets']['test'] = {}
        conf['training_sets']['test']['lr_train_dir'] = None
        conf['training_sets']['test']['hr_train_dir'] = None
        conf['training_sets']['test']['lr_valid_dir'] = None
        conf['training_sets']['test']['hr_valid_dir'] = None
        conf['loss_weights'] = None
        conf['training_sets']['test']['data_name'] = None
        conf['log_dirs'] = {}
        conf['log_dirs']['logs'] = None
        conf['log_dirs']['weights'] = None
        conf['weights_paths'] = {}
        conf['weights_paths']['generator'] = 'a/path/rdn-C1-D6-G1-G02-x0-weights.hdf5'
        conf['weights_paths']['discriminator'] = 'a/path/rdn-weights.hdf5'
        conf['session']['training']['n_validation_samples'] = None
        conf['session']['training']['metrics'] = None
        conf['session']['training']['learning_rate'] = {}
        conf['session']['training']['adam_optimizer'] = None
        conf['session']['training']['flatness'] = None
        conf['session']['training']['fallback_save_every_n_epochs'] = None
        conf['session']['training']['monitored_metrics'] = None
        conf['losses'] = None
        cls.conf = conf

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch('ISR.assistant._get_module', return_value=Object())
    @patch('ISR.train.trainer.Trainer', return_value=Object())
    def test_run_arguments_trainer(self, trainer, _get_module):
        with patch('yaml.load', return_value=self.conf):
            assistant.run(
                config_file='tests/data/config.yml', training=True, prediction=False, default=True
            )
            trainer.assert_called_once()

    @patch('ISR.assistant._get_module', return_value=Object())
    @patch('ISR.predict.predictor.Predictor', return_value=Object())
    def test_run_arguments_predictor(self, predictor, _get_module):
        with patch('yaml.load', return_value=self.conf):
            assistant.run(
                config_file='tests/data/config.yml', training=False, prediction=True, default=True
            )
            predictor.assert_called_once()
