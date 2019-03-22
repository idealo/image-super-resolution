import os
import argparse
import numpy as np
import yaml
from ISR.utils.logger import get_logger


logger = get_logger(__name__)


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', action='store_true', dest='prediction')
    parser.add_argument('--training', action='store_true', dest='training')
    parser.add_argument('--summary', action='store_true', dest='summary')
    parser.add_argument('--default', action='store_true', dest='default')
    parser.add_argument('--config', action='store', dest='config_file')
    return parser


def parse_args():
    """ Parse CLI arguments. """
    parser = _get_parser()
    args = vars(parser.parse_args())
    if args['prediction'] and args['training']:
        logger.error('Select only prediction OR training.')
        raise ValueError('Select only prediction OR training.')
    return args


def get_config_from_weights(w_path, arch_params, name):
    """
    Extracts architecture parameters from the file name of the weights.
    Only works with standardized weights name.
    """

    w_path = os.path.basename(w_path)
    parts = w_path.split(name)[1]
    parts = parts.split('_')[0]
    parts = parts.split('-')
    new_param = {}
    for param in arch_params:
        param_part = [x for x in parts if param in x]
        param_value = int(param_part[0].split(param)[1])
        new_param[param] = param_value
    return new_param


def select_option(options, message='', val=None):
    """ CLI selection given options. """

    while val not in options:
        val = input(message)
        if val not in options:
            logger.error('Invalid choice.')
    return val


def select_positive(message='', value=-1):
    """ CLI non-negative integer selection. """

    while value < 0:
        value = int(input(message))
        if value < 0:
            logger.error('Invalid choice.')
    return value


def browse_weights(weights_dir, model='generator'):
    """ Weights selection from cl. """

    exit = False
    while exit is False:
        weights = np.sort(os.listdir(weights_dir))[::-1]
        print_sel = dict(zip(np.arange(len(weights)), weights))
        for k in print_sel.keys():
            logger_message = '{item_n}: {item} \n'.format(item_n=k, item=print_sel[k])
            logger.info(logger_message)

        sel = select_positive('>>> Select folder or weights for {}\n'.format(model))
        if weights[sel].endswith('hdf5'):
            weights_path = os.path.join(weights_dir, weights[sel])
            exit = True
        else:
            weights_dir = os.path.join(weights_dir, weights[sel])
    return weights_path


def setup(config_file='config.yml', default=False, training=False, prediction=False):
    """CLI interface to set up the training or prediction session.

    Takes as input the configuration file path (minus the '.py' extension)
    and arguments parse from CLI.
    """

    conf = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    if training:
        session_type = 'training'
    elif prediction:
        session_type = 'prediction'
    else:
        message = '(t)raining or (p)rediction? (t/p) '
        session_type = {'t': 'training', 'p': 'prediction'}[select_option(['t', 'p'], message)]
    if default:
        all_default = 'y'
    else:
        all_default = input('Default options for everything? (y/[n]) ')

    if all_default == 'y':
        generator = conf['default']['generator']
        if session_type == 'prediction':
            dataset = conf['default']['test_set']
            conf['generators'][generator] = get_config_from_weights(
                conf['weights_paths']['generator'], conf['generators'][generator], generator
            )
        elif session_type == 'training':
            dataset = conf['default']['training_set']

        return session_type, generator, conf, dataset

    logger.info('Select SR (generator) network')
    generators = {}
    for i, gen in enumerate(conf['generators']):
        generators[str(i)] = gen
        logger.info('{}: {}'.format(i, gen))
    generator = generators[select_option(generators)]

    load_weights = input('Load pretrained weights for {}? ([y]/n/d) '.format(generator))
    if load_weights == 'n':
        default = input('Load default parameters for {}? ([y]/n) '.format(generator))
        if default == 'n':
            for param in conf['generators'][generator]:
                value = select_positive(message='{}:'.format(param))
                conf['generators'][generator][param] = value
        else:
            logger.info('Default {} parameters.'.format(generator))
    elif (load_weights == 'd') and (conf['weights_paths']['generator']):
        logger.info('Loading default weights for {}'.format(generator))
        logger.info(conf['weights_paths']['generator'])
        conf['generators'][generator] = get_config_from_weights(
            conf['weights_paths']['generator'], conf['generators'][generator], generator
        )
    else:
        conf['weights_paths']['generator'] = browse_weights(conf['dirs']['weights'], generator)
        conf['generators']['generator'] = get_config_from_weights(
            conf['weights_paths']['generator'], conf['generators'][generator], generator
        )
    logger.info('{} parameters:'.format(generator))
    logger.info(conf['generators'][generator])
    if session_type == 'training':
        use_discr = input('Use an Adversarial Network? (y/[n]) ')
        if use_discr == 'y':
            conf['default']['discriminator'] = True
            discr_w = input('Use pretrained discriminator weights? (y/[n]) ')
            if discr_w == 'y':
                conf['weights_paths']['discriminator'] = browse_weights(
                    conf['dirs']['weights'], 'discriminator'
                )

        use_feat_ext = input('Use feature extractor? (y/[n]) ')
        if use_feat_ext == 'y':
            conf['default']['feat_ext'] = True
    dataset = select_dataset(session_type, conf)

    return session_type, generator, conf, dataset


def select_dataset(session_type, conf):
    """ CLI snippet for selection the dataset for training. """

    if session_type == 'training':
        logger.info('Select training set')
        datasets = {}
        for i, data in enumerate(conf['training_sets']):
            datasets[str(i)] = data
            logger.info('{}: {}'.format(i, data))
        dataset = datasets[select_option(datasets)]

        return dataset
    else:
        logger.info('Select test set')
        datasets = {}
        for i, data in enumerate(conf['test_sets']):
            datasets[str(i)] = data
            logger.info('{}: {}'.format(i, data))
        dataset = datasets[select_option(datasets)]

        return dataset
