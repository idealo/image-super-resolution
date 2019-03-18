import os
import numpy as np
from importlib import import_module
from ISR.utils.utils import setup, parse_args
from ISR.utils.logger import get_logger


def _get_module(generator):
    return import_module('ISR.models.' + generator)


def run(config_file, default=False, training=False, prediction=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger = get_logger(__name__)
    session_type, generator, conf, dataset = setup(config_file, default, training, prediction)

    lr_patch_size = conf['session'][session_type]['patch_size']
    scale = conf['generators'][generator]['x']

    module = _get_module(generator)
    gen = module.make_model(conf['generators'][generator], lr_patch_size)
    if session_type == 'prediction':
        from ISR.predict.predictor import Predictor

        pr_h = Predictor(input_dir=conf['test_sets'][dataset])
        pr_h.get_predictions(gen, conf['weights_paths']['generator'])

    elif session_type == 'training':
        from ISR.train.trainer import Trainer

        hr_patch_size = lr_patch_size * scale
        if conf['default']['feat_ext']:
            from ISR.models.cut_vgg19 import Cut_VGG19

            out_layers = conf['feat_extr']['vgg19']['layers_to_extract']
            f_ext = Cut_VGG19(patch_size=hr_patch_size, layers_to_extract=out_layers)
        else:
            f_ext = None

        if conf['default']['discriminator']:
            from ISR.models.discriminator import Discriminator

            discr = Discriminator(patch_size=hr_patch_size, kernel_size=3)
        else:
            discr = None

        trainer = Trainer(
            generator=gen,
            discriminator=discr,
            feature_extractor=f_ext,
            lr_train_dir=conf['training_sets'][dataset]['lr_train_dir'],
            hr_train_dir=conf['training_sets'][dataset]['hr_train_dir'],
            lr_valid_dir=conf['training_sets'][dataset]['lr_valid_dir'],
            hr_valid_dir=conf['training_sets'][dataset]['hr_valid_dir'],
            loss_weights=conf['loss_weights'],
            dataname=conf['training_sets'][dataset]['data_name'],
            logs_dir=conf['dirs']['logs'],
            weights_dir=conf['dirs']['weights'],
            weights_generator=conf['weights_paths']['generator'],
            weights_discriminator=conf['weights_paths']['discriminator'],
            n_validation=conf['session'][session_type]['n_validation_samples'],
            lr_decay_frequency=conf['session'][session_type]['lr_decay_frequency'],
            lr_decay_factor=conf['session'][session_type]['lr_decay_factor'],
            T=0.01,
        )
        trainer.train(
            epochs=conf['session'][session_type]['epochs'],
            steps_per_epoch=conf['session'][session_type]['steps_per_epoch'],
            batch_size=conf['session'][session_type]['batch_size'],
        )

    else:
        logger.error('Invalid choice.')


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(1000)
    run(
        config_file=args['config_file'],
        default=args['default'],
        training=args['training'],
        prediction=args['prediction'],
    )
