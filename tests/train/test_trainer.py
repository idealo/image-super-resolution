import logging
import os
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

        def fake_folders(kind):
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

        with patch('os.listdir', side_effect=fake_folders):
            with patch('ISR.utils.datahandler.DataHandler._check_dataset', return_value=True):
                cls.trainer = Trainer(
                    generator=cls.RRDN,
                    discriminator=cls.discr,
                    feature_extractor=cls.f_ext,
                    lr_train_dir={'res': 'lr', 'matching': True},
                    hr_train_dir={'res': 'hr', 'matching': True},
                    lr_valid_dir={'res': 'lr', 'matching': True},
                    hr_valid_dir={'res': 'hr', 'matching': True},
                    learning_rate=0.0004,
                    loss_weights={'MSE': 1.0, 'discriminator': 1.0, 'feat_extr': 1.0},
                    logs_dir='./tests/temporary_test_data/logs',
                    weights_dir='./tests/temporary_test_data/weights',
                    dataname='TEST',
                    weights_generator=None,
                    weights_discriminator=None,
                    n_validation=2,
                    lr_decay_factor=0.5,
                    lr_decay_frequency=5,
                    T=0.01,
                )

    @classmethod
    def tearDownClass(cls):
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
        self.assertTrue(np.all(np.array(combined.loss_weights) == [1.0, 1.0, 0.5, 0.5]))
        mockd_trainer.discriminator = None
        combined = mockd_trainer._combine_networks()
        self.assertTrue(len(combined.layers) is 3)
        self.assertTrue(len(combined.loss_weights) is 3)
        self.assertTrue(np.all(np.array(combined.loss_weights) == [1.0, 0.5, 0.5]))
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

    def test_that_discriminator_and_f_extr_are_not_trainable_in_combined_model(self):
        combined = self.trainer._combine_networks()
        self.assertTrue(combined.get_layer('discriminator').trainable == False)
        self.assertTrue(combined.get_layer('feat_extr').trainable == False)

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
        mockd_trainer.train(epochs=1, steps_per_epoch=1, batch_size=1)
