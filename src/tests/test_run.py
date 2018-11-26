import os
import imageio
import argparse
import run
import models
from utils.utils import get_parser, load_model, load_configuration
from tests.common import TestWithData


class RunClassTest(TestWithData):
    def setUp(self):
        self.dataSetUp()
        pass

    def tearDown(self):
        self.dataTearDown()
        pass

    def make_model(self):
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

    def test_that_parser_returns_the_correct_namepace(self):
        parser = get_parser()
        cl_args = parser.parse_args(['--train'])
        namespace = cl_args._get_kwargs()
        self.assertTrue(('train', True) in namespace)
        self.assertTrue(('test', False) in namespace)


    def test_that_configuration_is_loaded_correctly(self):
        parser = get_parser()
        cl_args = parser.parse_args(['--train'])
        cl_args = vars(cl_args)
        load_configuration(cl_args, '../config.json')
        self.assertTrue(cl_args['G']==72)
        self.assertTrue(cl_args['log_dir']=='./logs')
        self.assertTrue(cl_args['weights_dir']=='./weights')
        self.assertTrue(cl_args['data_name']=='CUSTOM')


    def test_that_the_output_is_of_the_correct_scale(self):
        # create random weights
        self.make_model()
        self.weights_path = os.path.join(self.weights_folder, 'testweights_E001_.hdf5')
        self.rdn.save_weights(self.weights_path)
        # test the network with random weights
        self.create_random_dataset('correct', dataset_size=1)
        cl_args = ['--no_verbose', '--test']
        test_args = {'test_folder': self.dataset_folder['correct']['LR'],
                    'results_folder': self.results_folder,
                    'weights_path': self.weights_path}

        parser = get_parser()
        cl_args = parser.parse_args(cl_args)
        cl_args = vars(cl_args)
        load_configuration(cl_args, '../config.json')
        cl_args.update(test_args)
        cl_args.update(self.model_params)
        run.main(cl_args)
        for file in os.listdir(self.results_folder):
            img = imageio.imread(os.path.join(self.results_folder, file))
            self.assertTrue(img.shape == (self.img_size['HR'], self.img_size['HR'], 3))

    def test_if_train_is_executed(self):
        self.create_random_dataset('correct', dataset_size=14)
        train_arguments = {
                            'validation_labels': self.dataset_folder['correct']['HR'],
                            'validation_input': self.dataset_folder['correct']['LR'],
                            'training_labels': self.dataset_folder['correct']['HR'],
                            'training_input': self.dataset_folder['correct']['LR']
                           }
        cl_args = ['--pytest', '--no_verbose']
        parser = get_parser()
        cl_args = parser.parse_args(cl_args)
        cl_args = vars(cl_args)
        load_configuration(cl_args, '../config.json')
        cl_args.update(train_arguments)
        run.main(cl_args)
        self.assertTrue(1!=0)
