import os
import numpy as np
from ISR.utils.logger import get_logger


class TrainerHelper:
    """Collection of useful functions to manage training sessions.

    Args:
        generator: Keras model, the super-scaling, or generator, network.
        logs_dir: path to the directory where the tensorboard logs are saved.
        weights_dir: path to the directory where the weights are saved.
        lr_train_dir: path to the directory containing the Low-Res images.
        feature_extractor: Keras model, feature extractor network for the deep features
            component of perceptual loss function.
        discriminator: Keras model, the discriminator network for the adversarial
            component of the perceptual loss.
        dataname: string, used to identify what dataset is used for the training session.
        pretrained_weights_path: dictionary, contains the paths, if any to the
            pre-trained generator's and to the pre-trained discriminator's weights,
            for transfer learning.
        fallback_save_every_n_epochs: integer, determines after how many epochs that did not trigger
            weights saving the weights are despite no metric improvement.


    Methods:
        print_training_setting: see docstring.
        on_epoch_end: see docstring.
        epoch_n_from_weights_name: see docstring.
        initialize_training: see docstring.

    """

    def __init__(
        self,
        generator,
        weights_dir,
        logs_dir,
        lr_train_dir,
        feature_extractor=None,
        discriminator=None,
        dataname=None,
        pretrained_weights_path={},
        fallback_save_every_n_epochs=2,
    ):
        self.generator = generator
        self.dirs = {'logs': logs_dir, 'weights': weights_dir}
        self.feature_extractor = feature_extractor
        self.discriminator = discriminator
        self.dataname = dataname
        self.pretrained_weights_path = pretrained_weights_path
        self.lr_dir = lr_train_dir
        self.best_metrics = {}
        self.fallback_save_every_n_epochs = fallback_save_every_n_epochs
        self.since_last_epoch = 0
        self.logger = get_logger(__name__)

    def _make_generator_name(self):
        """ Combines generators's name and its architecture's parameters. """

        gen_name = self.generator.name
        params = [gen_name]
        for param in np.sort(list(self.generator.params.keys())):
            params.append('{g}{p}'.format(g=param, p=self.generator.params[param]))
        return '-'.join(params)

    def _make_basename(self):
        """
        Constructs the name of the training session from training settings
        and the pre-trained weights, if any.
        """

        current_training_setup = self.dataname
        if self.feature_extractor:
            params = [self.feature_extractor.name]
            for feat in self.feature_extractor.layers_to_extract:
                params.append(str(feat))
            feat_name = '-'.join(params)
            current_training_setup = '-'.join([current_training_setup, feat_name])
        if self.discriminator:
            current_training_setup = '-'.join([current_training_setup, self.discriminator.name])

        # append to old description if any, otherwise append to gen architecture (which is constant across trainings)
        if ('generator' in self.pretrained_weights_path) and self.pretrained_weights_path[
            'generator'
        ]:
            root = os.path.splitext(os.path.basename(self.pretrained_weights_path['generator']))[0]
        else:
            root = self.generator_description
        return '_'.join([root, current_training_setup])

    def _get_basepath(self):
        """ Creates the dir structure needed to save weights and logs from the basename. """

        subdirs = self.basename.split('_')
        return os.path.join(*subdirs)

    def _create_dirs(self):
        """ Creates the directories associated with the class instance. """

        for dir in self.dirs:
            if not os.path.exists(os.path.join(self.dirs[dir], self.basepath)):
                os.makedirs(os.path.join(self.dirs[dir], self.basepath), exist_ok=False)

    def _make_callback_paths(self):
        """ Creates the paths used for managing logs and weights storage. """

        callback_paths = {}
        callback_paths['weights'] = os.path.join(self.dirs['weights'], self.basepath)
        callback_paths['logs'] = os.path.join(self.dirs['logs'], self.basepath)
        return callback_paths

    def _weights_name(self):
        """ Builds the string used to name the weights of the training session. """

        w_name = {'generator': '-'.join([self.basename, 'e{epoch:03d}.hdf5'])}
        if self.discriminator:
            w_name.update(
                {'discriminator': '-'.join(['discr', self.basename, 'e{epoch:03d}.hdf5'])}
            )
        return w_name

    def print_training_setting(self):
        """ Does what it says. """

        self.logger.info('\nTraining details:')
        self.logger.info('Generator settings:\n{}'.format(self.generator.params))
        if self.pretrained_weights_path['generator']:
            self.logger.info('Generator pretrained weights:')
            self.logger.info(
                '>> {}'.format(os.path.basename(self.pretrained_weights_path['generator']))
            )
        if self.discriminator:
            self.logger.info('Using GAN discriminator.')
            if self.pretrained_weights_path['discriminator']:
                self.logger.info('Discriminator pretrained weights:')
                self.logger.info(
                    '>> {}'.format(os.path.basename(self.pretrained_weights_path['discriminator']))
                )
        if self.feature_extractor:
            self.logger.info('Using high level features loss:')
            self.logger.info(
                '{f_ext} layers: {layers}'.format(
                    f_ext=self.feature_extractor.model.name,
                    layers=self.feature_extractor.layers_to_extract,
                )
            )
        self.logger.info('Training session name identifier: {}'.format(self.basename))
        self.logger.info('Input dir: {}'.format(self.lr_dir))
        self.logger.info('Patch size: {}'.format(self.generator.patch_size))
        self.logger.info('Saving weights under: {}'.format(self.callback_paths['weights']))
        self.logger.info('Saving tensorboard logs under: {}'.format(self.callback_paths['logs']))

    def _save_weights(self, epoch, generator, discriminator=None, best=False):
        """ Saves the weights of the non-None models. """

        if best:
            w_name = 'BEST-' + self.weights_name['generator'].format(epoch=epoch + 1)
        else:
            w_name = self.weights_name['generator'].format(epoch=epoch + 1)
        gen_path = os.path.join(self.callback_paths['weights'], w_name)
        generator.save_weights(
            gen_path
        )  # CANT SAVE MODEL DUE TO TF LAYER INSIDE LAMBDA (PIXELSHUFFLE)
        if discriminator:
            if best:
                w_name = 'BEST-' + self.weights_name['discriminator'].format(epoch=epoch + 1)
            else:
                w_name = self.weights_name['discriminator'].format(epoch=epoch + 1)
            discr_path = os.path.join(self.callback_paths['weights'], w_name)
            discriminator.model.save_weights(discr_path)
        try:
            self._remove_old_weights(5)
        except Exception as e:
            self.logger.warning('Could not remove weights'.format(e))

    def _remove_old_weights(self, max_n_weights, max_best=3):
        """Scans the weights folder and removes all but:
            - the max_best newest 'best' weights.
            - max_n_weights most recent 'others' weights.
        """

        w_list = {}
        w_list['all'] = [
            w for w in os.listdir(self.callback_paths['weights']) if w.endswith('.hdf5')
        ]
        w_list['best'] = [w for w in w_list['all'] if w.startswith('BEST')]
        w_list['others'] = [w for w in w_list['all'] if w not in w_list['best']]
        # remove older best
        epochs_set = {}
        epochs_set['best'] = list(set([self.epoch_n_from_weights_name(w) for w in w_list['best']]))
        epochs_set['others'] = list(
            set([self.epoch_n_from_weights_name(w) for w in w_list['others']])
        )
        keep_max = {'best': max_best, 'others': max_n_weights}
        for type in ['others', 'best']:
            if len(epochs_set[type]) > keep_max[type]:
                epoch_list = np.sort(epochs_set[type])[::-1]
                epoch_list = epoch_list[0 : keep_max[type]]
                for w in w_list[type]:
                    if self.epoch_n_from_weights_name(w) not in epoch_list:
                        # print('remove', os.path.join(self.callback_paths['weights'], w))
                        os.remove(os.path.join(self.callback_paths['weights'], w))

    def on_epoch_end(
        self, epoch, losses, generator, discriminator=None, metrics={'val_generator_loss': 'min'}
    ):
        """
        Manages the operations that are taken at the end of each epoch:
        metric checks, weight saves, logging.
        """

        self.logger.info(losses)
        monitor_op = {'max': np.greater, 'min': np.less}
        extreme = {'max': -np.Inf, 'min': np.Inf}
        for metric in metrics:
            if metric in losses.keys():
                if metric not in self.best_metrics.keys():
                    self.best_metrics[metric] = extreme[metrics[metric]]

                if monitor_op[metrics[metric]](losses[metric], self.best_metrics[metric]):
                    self.logger.info(
                        '{} improved from {:10.5f} to {:10.5f}'.format(
                            metric, self.best_metrics[metric], losses[metric]
                        )
                    )
                    self.logger.info('Saving weights')
                    self.best_metrics[metric] = losses[metric]
                    self._save_weights(epoch, generator, discriminator, best=True)
                    self.since_last_epoch = 0
                    return True
                else:
                    self.logger.info('{} did not improve.'.format(metric))
                    if self.since_last_epoch >= self.fallback_save_every_n_epochs:
                        self.logger.info('Saving weights anyways.')
                        self._save_weights(epoch, generator, discriminator, best=False)
                        self.since_last_epoch = 0
                        return True

            else:
                self.logger.warning('{} is not monitored, cannot save weights.'.format(metric))
        self.since_last_epoch += 1
        return False

    def epoch_n_from_weights_name(self, w_name):
        """
        Extracts the last epoch number from the standardized weights name.
        Only works with standardized weights names.
        """

        return int(os.path.splitext(os.path.basename(w_name))[0].split('-')[-1][1:])

    def initialize_training(self, object):
        """Function that is exectured prior to training.

        Wraps up most of the functions of this class:
        load the weights if any are given, generaters names for session and weights,
        creates directories and prints the training session.
        """
        object.pretrained_weights_path = self.pretrained_weights_path
        object._load_weights()
        w_name = object.pretrained_weights_path['generator']
        if w_name:
            last_epoch = self.epoch_n_from_weights_name(w_name)
        else:
            last_epoch = 0

        self.generator_description = self._make_generator_name()
        self.basename = self._make_basename()
        self.basepath = self._get_basepath()
        self.weights_name = self._weights_name()
        self.callback_paths = self._make_callback_paths()
        self._create_dirs()
        self.print_training_setting()
        return last_epoch
