import os
import json
import argparse
import numpy as np
from imageio import imread
from importlib import import_module
from logging import info, error, debug, warning


def load_model(model_parameters, add_vgg=False, model_name='rdn', verbose='True'):
    if add_vgg:
        model_name = '_'.join([model_name, 'vgg'])

    info('\n>>> Creating the {} network'.format(model_name.upper()))

    module = import_module('models.' + model_name.lower())
    model = module.make_model(**model_parameters)
    return model


def browse_weights(weights_dir='./weights', weights_path=''):
    """Weight slection function.
    IF a selection is made, returns last epoch's number
    """
    if weights_path is not '':
        _, weights_name = os.path.split(weights_path)
        last_epoch = int(weights_name.split('_')[1][1:])
        return weights_path, last_epoch
    else:
        print('\n>>> Existing weights \n')
        exit = False
        while exit is False:
            weights = np.sort(os.listdir(weights_dir))[::-1]
            print_sel = dict(zip(np.arange(len(weights)), weights))
            for k in print_sel.keys():
                print(k, ':', print_sel[k], '\n')

            sel = int(input('>>> Select weights or folder (-1 to skip)\n'))
            # Weight file name:
            # weights<dataname>_E<epoch_#>_X<scale>_N<#_training_samples>.hdf5
            if sel > -1:
                if weights[sel].endswith('hdf5'):
                    last_epoch = int(weights[sel].split('_')[1][1:])
                    return os.path.join(weights_dir, weights[sel]), last_epoch
                else:
                    weights_dir = os.path.join(weights_dir, weights[sel])

            else:
                exit = True
    return None, 0


def load_configuration(args, config_file='./config.json'):
    json_dict = json.load(open(config_file))
    args.update(json_dict['model parameters'])
    args.update(json_dict['folders']['weights'])
    if args['train']:
        args.update(json_dict['train'])
        args.update(json_dict['folders']['custom data'])
    if args['div2k']:
        args.update(json_dict['folders']['div2k'])
    if args['aws-div2k']:
        args.update(json_dict['folders']['aws-div2k'])
    if args['test']:
        args.update(json_dict['test'])
        args.update(json_dict['folders']['test'])
    if args['pre-trained']:
        args.update(json_dict['pre-trained'])
    if args['pytest']:
        args.update(json_dict['pytest'])
        

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', dest='test')
    parser.add_argument('--pytest', action='store_true', dest='pytest')
    parser.add_argument('--train', action='store_true', dest='train')
    parser.add_argument('--div2k', action='store_true', dest='div2k')
    parser.add_argument('--aws-div2k', action='store_true', dest='aws-div2k')
    parser.add_argument('--summary', action='store_true', dest='summary')
    parser.add_argument('--custom-data', action='store_true', dest='custom-data')
    parser.add_argument('--pre-trained', action='store_true', dest='pre-trained')
    parser.add_argument('-nv', '--no_verbose', action='store_false', dest='verbose')
    return parser
