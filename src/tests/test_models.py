import os
import numpy as np
from trainer.train import Trainer
from utils.utils import load_model, get_parser, load_configuration
from tests.common import TestWithData


class ModelsClassTest(TestWithData):
    def setUp(self):
        self.model_params = {'learning_rate': 1e-5,
                             'kernel_size': 3,
                             'scale': 2,
                             'c_dim': 3,
                             'G0': 25,
                             'G': 50,
                             'D': 5,
                             'C': 3}

        self.model = load_model(self.model_params, add_vgg=False, model_name='rdn', verbose=False)
        self.rdn = self.model.rdn
        self.dataSetUp()
        pass

    def tearDown(self):
        self.dataTearDown()

    def RDB_param_number(self):
        map_size = self.model_params['kernel_size'] ** 2
        G = self.model_params['G']
        C = self.model_params['C']
        G0 = self.model_params['G0']

        params = {}
        for c in range(C):
            params['F_x_%d' % (c + 1)] = map_size * G * (G0 + c * G) + G
        params['LFF_x'] = G0 * (C * G + G0) + G0
        return params

    def test_number_of_parameters_per_layer(self):
        map_size = self.model_params['kernel_size'] ** 2
        D = self.model_params['D']
        G = self.model_params['G']
        C = self.model_params['C']
        G0 = self.model_params['G0']
        ch = self.model_params['c_dim']
        scale = self.model_params['scale']
        RDB_params = self.RDB_param_number()

        params = {}
        for d in range(1, D + 1):
            for c in range(1, C + 1):
                params['F_%d_%d' % (d, c)] = RDB_params['F_x_%d' % (c)]
            params['LFF_%d' % (d)] = RDB_params['LFF_x']
        params['F_m1'] = map_size * ch * G0 + G0
        params['F_0'] = map_size * G0 * G0 + G0
        params['GFF_1'] = D * G0 * G0 + G0
        params['GFF_2'] = map_size * G0 * G0 + G0
        params['UPN1'] = 25 * G0 * 64 + 64
        params['UPN2'] = 9 * 64 * 32 + 32
        params['UPN3'] = (ch * scale ** 2) * 9 * 32 + ch * scale ** 2
        params['SR'] = ch * (ch * scale ** 2) * map_size + ch
        for layer in self.rdn.layers:
            layer_weights = layer.get_weights()
            layer_name = layer.get_config()['name']
            if len(layer_weights) > 0:
                if not layer_name in params.keys():
                    raise
                layer_param = np.prod(layer_weights[0].shape) + layer_weights[1].shape[0]
                self.assertEqual(layer_param, params[layer_name],
                                 'Weight number mismatch at layer %s' % (layer_name))

        tot_param = sum([params[k] for k in params.keys()])
        self.assertEqual(tot_param, self.rdn.count_params(),
                         'Total parameters number mismatch')

    def test_layer_output_shapes(self):
        map_size = self.model_params['kernel_size'] ** 2
        D = self.model_params['D']
        G = self.model_params['G']
        C = self.model_params['C']
        G0 = self.model_params['G0']
        ch = self.model_params['c_dim']
        scale = self.model_params['scale']

        shape = {}
        shape['LR'] = (None, None, None, ch)
        shape['F_m1'] = (None, None, None, G0)
        shape['F_0'] = (None, None, None, G0)
        for d in range(1, D + 1):
            for c in range(1, C + 1):
                shape['F_%d_%d' % (d, c)] = (None, None, None, G)
                shape['F_%d_%d_Relu' % (d, c)] = (None, None, None, G)
                shape['RDB_Concat_%d_%d' % (d, c)] = (None, None, None, c * G + G0)
            shape['LFF_%d' % (d)] = (None, None, None, G0)
            shape['LRL_%d' % (d)] = (None, None, None, G0)
        shape['LRLs_Concat'] = (None, None, None, G0 * D)
        shape['GFF_1'] = (None, None, None, G0)
        shape['GFF_2'] = (None, None, None, G0)
        shape['FDF'] = (None, None, None, G0)
        shape['UPN1'] = (None, None, None, 64)
        shape['UPN1_Relu'] = (None, None, None, 64)
        shape['UPN2'] = (None, None, None, 32)
        shape['UPN2_Relu'] = (None, None, None, 32)
        shape['UPN3'] = (None, None, None, ch * scale ** 2)
        shape['UPsample'] = (None, None, None, ch * scale ** 2)
        shape['SR'] = (None, None, None, ch)

        for layer in self.rdn.layers:
            layer_shape = layer.output_shape
            layer_name = layer.get_config()['name']
            if not layer_name in shape.keys():
                raise
            self.assertEqual(layer_shape, shape[layer_name],
                             'Shape mismatch at layer %s' % (layer_name))

    def test_if_trainable_weights_update_with_one_step(self):
        self.scale = self.model_params['scale']
        self.img_size = {'HR': 10 * self.scale,
                         'LR': 10}
        self.dataset_size = 8
        self.create_random_dataset(type='correct')

        before_step = []
        for layer in self.model.rdn.layers:
            if len(layer.trainable_weights) > 0:
                before_step.append(layer.get_weights()[0])

        train_arguments = {
                            'validation_labels': self.dataset_folder['correct']['HR'],
                            'validation_input': self.dataset_folder['correct']['LR'],
                            'training_labels': self.dataset_folder['correct']['HR'],
                            'training_input': self.dataset_folder['correct']['LR'],
                           }
        cl_args = ['--pytest', '--no_verbose']
        parser = get_parser()
        cl_args = parser.parse_args(cl_args)
        cl_args = vars(cl_args)
        load_configuration(cl_args, '../config.json')
        cl_args.update(train_arguments)
        trainer = Trainer(train_arguments=cl_args)

        i = 0
        for layer in self.model.rdn.layers:
            if len(layer.trainable_weights) > 0:
                self.assertTrue(np.all(before_step[i] == layer.get_weights()[0]))
                i += 1

        trainer.train_model(self.model)

        i = 0
        for layer in self.model.rdn.layers:
            if len(layer.trainable_weights) > 0:
                self.assertFalse(np.all(before_step[i] == layer.get_weights()[0]))
                i += 1
