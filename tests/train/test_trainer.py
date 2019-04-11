import logging
import os
import shutil
from pathlib import Path
import unittest
import yaml
import numpy as np
from copy import copy
from ISR.models.cut_vgg19 import Cut_VGG19
from ISR.models.discriminator import Discriminator
from ISR.models.rrdn import RRDN
from ISR.train.trainer import Trainer
from unittest.mock import patch, Mock


class TrainerClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = yaml.load(open(os.path.join('tests', 'data', 'config.yml'), 'r'))
        cls.RRDN = RRDN(arch_params=cls.setup['rrdn'], patch_size=cls.setup['patch_size'])
        cls.f_ext = Cut_VGG19(patch_size=cls.setup['patch_size'] * 2, layers_to_extract=[1, 2])
        cls.discr = Discriminator(patch_size=cls.setup['patch_size'] * 2)
        cls.weights_path = {
            'generator': os.path.join(cls.setup['weights_dir'], 'test_gen_weights.hdf5'),
            'discriminator': os.path.join(cls.setup['weights_dir'], 'test_dis_weights.hdf5'),
        }
        cls.temp_data = Path('tests/temporary_test_data')

        cls.not_matching_hr = cls.temp_data / 'not_matching_hr'
        cls.not_matching_hr.mkdir(parents=True)
        for item in ['data2.gif', 'data1.png', 'data0.jpeg']:
            (cls.not_matching_hr / item).touch()

        cls.not_matching_lr = cls.temp_data / 'not_matching_lr'
        cls.not_matching_lr.mkdir(parents=True)
        for item in ['data1.png']:
            (cls.not_matching_lr / item).touch()

        cls.matching_hr = cls.temp_data / 'matching_hr'
        cls.matching_hr.mkdir(parents=True)
        for item in ['data2.gif', 'data1.png', 'data0.jpeg']:
            (cls.matching_hr / item).touch()

        cls.matching_lr = cls.temp_data / 'matching_lr'
        cls.matching_lr.mkdir(parents=True)
        for item in ['data1.png', 'data0.jpeg']:
            (cls.matching_lr / item).touch()

        with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
            cls.trainer = Trainer(
                generator=cls.RRDN,
                discriminator=cls.discr,
                feature_extractor=cls.f_ext,
                lr_train_dir=str(cls.matching_lr),
                hr_train_dir=str(cls.matching_hr),
                lr_valid_dir=str(cls.matching_lr),
                hr_valid_dir=str(cls.matching_hr),
                learning_rate={'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 5},
                log_dirs={
                    'logs': './tests/temporary_test_data/logs',
                    'weights': './tests/temporary_test_data/weights',
                },
                dataname='TEST',
                weights_generator=None,
                weights_discriminator=None,
                n_validation=2,
                flatness={'min': 0.01, 'max': 0.3, 'increase': 0.01, 'increase_frequency': 5},
                adam_optimizer={'beta1': 0.9, 'beta2': 0.999, 'epsilon': None},
                losses={'generator': 'mae', 'discriminator': 'mse', 'feature_extractor': 'mse'},
                loss_weights={'generator': 1.0, 'discriminator': 1.0, 'feature_extractor': 0.5},
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_data)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test__combine_networks_sanity(self):
        mockd_trainer = copy(self.trainer)
        combined = mockd_trainer._combine_networks()
        self.assertTrue(len(combined.layers) is 4)
        self.assertTrue(len(combined.loss_weights) is 4)
        self.assertTrue(np.all(np.array(combined.loss_weights) == [1.0, 1.0, 0.25, 0.25]))
        mockd_trainer.discriminator = None
        combined = mockd_trainer._combine_networks()
        self.assertTrue(len(combined.layers) is 3)
        self.assertTrue(len(combined.loss_weights) is 3)
        self.assertTrue(np.all(np.array(combined.loss_weights) == [1.0, 0.25, 0.25]))
        mockd_trainer.feature_extractor = None
        combined = mockd_trainer._combine_networks()
        self.assertTrue(len(combined.layers) is 2)
        self.assertTrue(len(combined.loss_weights) is 1)
        self.assertTrue(np.all(np.array(combined.loss_weights) == [1.0]))
        try:
            mockd_trainer.generator = None
            combined = mockd_trainer._combine_networks()
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test__lr_scheduler(self):
        lr = self.trainer._lr_scheduler(epoch=10)
        expected_lr = 0.0004 * (0.5) ** 2
        self.assertTrue(lr == expected_lr)

    def test__flatness_scheduler(self):
        # test with arguments values
        f = self.trainer._flatness_scheduler(epoch=10)
        expected_flatness = 0.03
        self.assertTrue(f == expected_flatness)

        # test with specified values
        self.trainer.flatness['increase'] = 0.1
        self.trainer.flatness['increase_frequency'] = 2
        self.trainer.flatness['min'] = 0.1
        self.trainer.flatness['max'] = 1.0
        f = self.trainer._flatness_scheduler(epoch=10)
        expected_flatness = 0.6
        self.assertTrue(f == expected_flatness)

        # test max
        self.trainer.flatness['increase'] = 1.0
        self.trainer.flatness['increase_frequency'] = 1
        self.trainer.flatness['min'] = 0.1
        self.trainer.flatness['max'] = 1.0
        f = self.trainer._flatness_scheduler(epoch=10)
        expected_flatness = 1.0
        self.assertTrue(f == expected_flatness)

    def test_that_discriminator_and_f_extr_are_not_trainable_in_combined_model(self):
        combined = self.trainer._combine_networks()
        self.assertTrue(combined.get_layer('discriminator').trainable == False)
        self.assertTrue(combined.get_layer('feature_extractor').trainable == False)

    def test_that_discriminator_is_trainable_outside_of_combined(self):
        combined = self.trainer._combine_networks()
        y = np.random.random((1, self.setup['patch_size'] * 2, self.setup['patch_size'] * 2, 3))
        discr_out_shape = list(self.discr.model.outputs[0].shape)[1:4]
        valid = np.ones([1] + discr_out_shape)

        before_step = []
        for layer in self.trainer.discriminator.model.layers:
            if len(layer.trainable_weights) > 0:
                before_step.append(layer.get_weights()[0])

        self.trainer.discriminator.model.train_on_batch(y, valid)

        i = 0
        for layer in self.trainer.discriminator.model.layers:
            if len(layer.trainable_weights) > 0:
                self.assertFalse(np.all(before_step[i] == layer.get_weights()[0]))
                i += 1

    def test_that_feature_extractor_is_not_trainable_outside_of_combined(self):
        mockd_trainer = copy(self.trainer)
        y = np.random.random((1, self.setup['patch_size'] * 2, self.setup['patch_size'] * 2, 3))
        f_ext_out_shape = list(mockd_trainer.feature_extractor.model.outputs[0].shape[1:4])
        f_ext_out_shape1 = list(mockd_trainer.feature_extractor.model.outputs[1].shape[1:4])
        feats = [np.random.random([1] + f_ext_out_shape), np.random.random([1] + f_ext_out_shape1)]
        # should not have optimizer
        try:
            mockd_trainer.feature_extractor.model.train_on_batch(y, [*feats])
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test__load_weights(self):
        def check_gen_path(path):
            self.assertTrue(path == 'gen')

        def check_discr_path(path):
            self.assertTrue(path == 'discr')

        mockd_trainer = copy(self.trainer)

        mockd_trainer.pretrained_weights_path = {'generator': 'gen', 'discriminator': 'discr'}
        mockd_trainer.discriminator.model.load_weights = Mock(side_effect=check_discr_path)
        mockd_trainer.model.get_layer('generator').load_weights = Mock(side_effect=check_gen_path)
        mockd_trainer._load_weights()

    def test_train(self):
        def nullifier(*args):
            pass

        mockd_trainer = copy(self.trainer)
        mockd_trainer.logger = Mock(side_effect=nullifier)
        mockd_trainer.valid_dh.get_validation_set = Mock(return_value={'lr': [], 'hr': []})
        mockd_trainer.train_dh.get_batch = Mock(return_value={'lr': [], 'hr': []})
        mockd_trainer.feature_extractor.model.predict = Mock(return_value=[])
        mockd_trainer.generator.model.predict = Mock(return_value=[])
        mockd_trainer.discriminator.model.train_on_batch = Mock(return_value=[])
        mockd_trainer.model.train_on_batch = Mock(return_value=[])
        mockd_trainer.model.evaluate = Mock(return_value=[])
        mockd_trainer.tensorboard = Mock(side_effect=nullifier)
        mockd_trainer.helper.on_epoch_end = Mock(return_value=True)

        logging.disable(logging.CRITICAL)
        mockd_trainer.train(epochs=1, steps_per_epoch=1, batch_size=1, monitored_metrics={})
