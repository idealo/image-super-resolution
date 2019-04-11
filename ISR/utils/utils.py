import os
import argparse
from datetime import datetime
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


def get_timestamp():
    ts = datetime.now()
    time_stamp = '{y}-{m:02d}-{d:02d}_{h:02d}:{mm:02d}'.format(
        y=ts.year, m=ts.month, d=ts.day, h=ts.hour, mm=ts.minute
    )
    return time_stamp


def check_parameter_keys(parameter, needed_keys, optional_keys=None, default_value=None):
    if needed_keys:
        for key in needed_keys:
            if key not in parameter:
                logger.error('{p} is missing key {k}'.format(p=parameter, k=key))
                raise
    if optional_keys:
        for key in optional_keys:
            if key not in parameter:
                logger.info('Setting {k} in {p} to {d}'.format(k=key, p=parameter, d=default_value))
                parameter[key] = default_value


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


def select_multiple_options(options, message='', val=None):
    """ CLI multiple selection given options. """

    n_options = len(options)
    valid_selections = False
    selected_options = []
    while not valid_selections:
        for i, opt in enumerate(np.sort(options)):
            logger.info('{}: {}'.format(i, opt))
        val = input(message + ' (space separated selection)\n')
        vals = val.split(' ')
        valid_selections = True
        for v in vals:
            if int(v) not in list(range(n_options)):
                logger.error('Invalid choice.')
                valid_selections = False
            else:
                selected_options.append(options[int(v)])

    return selected_options


def select_bool(message=''):
    """ CLI bool selection. """

    options = ['y', 'n']
    message = message + ' (' + '/'.join(options) + ') '
    val = None
    while val not in options:
        val = input(message)
        if val not in options:
            logger.error('Input y (yes) or n (no).')
    if val == 'y':
        return True
    elif val == 'n':
        return False


def select_positive_float(message=''):
    """ CLI non-negative float selection. """

    value = -1
    while value < 0:
        value = float(input(message))
        if value < 0:
            logger.error('Invalid choice.')
    return value


def select_positive_integer(message='', value=-1):
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

        sel = select_positive_integer('>>> Select folder or weights for {}\n'.format(model))
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
        all_default = select_bool('Default options for everything?')

    if all_default:
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
        default = select_bool('Load default parameters for {}?'.format(generator))
        if not default:
            for param in conf['generators'][generator]:
                value = select_positive_integer(message='{}:'.format(param))
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
        default_loss_weights = select_bool('Use default weights for loss components?')
        if not default_loss_weights:
            conf['loss_weights']['generator'] = select_positive_float(
                'Input coefficient for pixel-wise generator loss component '
            )
        use_discr = select_bool('Use an Adversarial Network?')
        if use_discr:
            conf['default']['discriminator'] = True
            discr_w = select_bool('Use pretrained discriminator weights?')
            if discr_w:
                conf['weights_paths']['discriminator'] = browse_weights(
                    conf['dirs']['weights'], 'discriminator'
                )
            if not default_loss_weights:
                conf['loss_weights']['discriminator'] = select_positive_float(
                    'Input coefficient for Adversarial loss component '
                )

        use_feature_extractor = select_bool('Use feature extractor?')
        if use_feature_extractor:
            conf['default']['feature_extractor'] = True
            if not default_loss_weights:
                conf['loss_weights']['feature_extractor'] = select_positive_float(
                    'Input coefficient for conv features loss component '
                )
        default_metrics = select_bool('Monitor default metrics?')
        if not default_metrics:
            suggested_list = suggest_metrics(use_discr, use_feature_extractor)
            selected_metrics = select_multiple_options(
                list(suggested_list.keys()), message='Select metrics to monitor.'
            )

            conf['session']['training']['monitored_metrics'] = {}
            for metric in selected_metrics:
                conf['session']['training']['monitored_metrics'][metric] = suggested_list[metric]
            print(conf['session']['training']['monitored_metrics'])

    dataset = select_dataset(session_type, conf)

    return session_type, generator, conf, dataset


def suggest_metrics(discriminator=False, feature_extractor=False, loss_weights={}):
    suggested_metrics = {}
    if not discriminator and not feature_extractor:
        suggested_metrics['val_loss'] = 'min'
        suggested_metrics['train_loss'] = 'min'
        suggested_metrics['val_PSNR'] = 'max'
        suggested_metrics['train_PSNR'] = 'max'
    if feature_extractor or discriminator:
        suggested_metrics['val_generator_loss'] = 'min'
        suggested_metrics['train_generator_loss'] = 'min'
        suggested_metrics['val_generator_PSNR'] = 'max'
        suggested_metrics['train_generator_PSNR'] = 'max'
    if feature_extractor:
        suggested_metrics['val_feature_extractor_loss'] = 'min'
        suggested_metrics['train_feature_extractor_loss'] = 'min'
    return suggested_metrics


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
