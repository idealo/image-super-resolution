import yaml
import numpy as np
from pathlib import Path
from ISR.utils.logger import get_logger
from ISR.utils.utils import get_timestamp


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
        weights_dictionarycontains the paths, if any to the
            pre-trained generator's and to the pre-trained discriminator's weights,
            for transfer learning.
        fallback_save_every_n_epochs: integer, determines after how many epochs that did not trigger
            weights saving the weights are despite no metric improvement.
        max_n_best_weights: maximum amount of weights that are best on some metric that are kept.
        max_n_other_weights: maximum amount of non-best weights that are kept.


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
        weights_generator=None,
        weights_discriminator=None,
        fallback_save_every_n_epochs=2,
        max_n_other_weights=5,
        max_n_best_weights=5,
    ):
        self.generator = generator
        self.dirs = {'logs': Path(logs_dir), 'weights': Path(weights_dir)}
        self.feature_extractor = feature_extractor
        self.discriminator = discriminator
        self.dataname = dataname

        if weights_generator:
            self.pretrained_generator_weights = Path(weights_generator)
        else:
            self.pretrained_generator_weights = None

        if weights_discriminator:
            self.pretrained_discriminator_weights = Path(weights_discriminator)
        else:
            self.pretrained_discriminator_weights = None

        self.fallback_save_every_n_epochs = fallback_save_every_n_epochs
        self.lr_dir = Path(lr_train_dir)
        self.basename = self._make_basename()
        self.session_id = self.get_session_id(basename=None)
        self.session_config_name = 'session_config.yml'
        self.callback_paths = self._make_callback_paths()
        self.weights_name = self._weights_name(self.callback_paths)
        self.best_metrics = {}
        self.since_last_epoch = 0
        self.max_n_other_weights = max_n_other_weights
        self.max_n_best_weights = max_n_best_weights
        self.logger = get_logger(__name__)

    def _make_basename(self):
        """ Combines generators's name and its architecture's parameters. """

        gen_name = self.generator.name
        params = [gen_name]
        for param in np.sort(list(self.generator.params.keys())):
            params.append('{g}{p}'.format(g=param, p=self.generator.params[param]))
        return '-'.join(params)

    def get_session_id(self, basename):
        """ Returns unique session identifier. """

        time_stamp = get_timestamp()

        if basename:
            session_id = '{b}_{ts}'.format(b=basename, ts=time_stamp)
        else:
            session_id = time_stamp
        return session_id

    def _get_previous_conf(self):
        """ Checks if a session_config.yml is available in the pretrained weights folder. """

        if self.pretrained_generator_weights:
            session_config_path = (
                self.pretrained_generator_weights.parent / self.session_config_name
            )
            if session_config_path.exists():
                return yaml.load(session_config_path.read_text(), Loader=yaml.FullLoader)
            else:
                self.logger.warning('Could not find previous configuration')
                return {}

        return {}

    def update_config(self, training_settings):
        """
        Adds to the existing settings (if any) the current settings dictionary
        under the session_id key.
        """

        session_settings = self._get_previous_conf()
        session_settings.update({self.session_id: training_settings})

        return session_settings

    def _make_callback_paths(self):
        """ Creates the paths used for managing logs and weights storage. """

        callback_paths = {}
        callback_paths['weights'] = self.dirs['weights'] / self.basename / self.session_id
        callback_paths['logs'] = self.dirs['logs'] / self.basename / self.session_id
        return callback_paths

    def _weights_name(self, callback_paths):
        """ Builds the string used to name the weights of the training session. """

        w_name = {
            'generator': callback_paths['weights']
            / (self.basename + '{metric}_epoch{epoch:03d}.hdf5')
        }
        if self.discriminator:
            w_name.update(
                {
                    'discriminator': callback_paths['weights']
                    / (self.discriminator.name + '{metric}_epoch{epoch:03d}.hdf5')
                }
            )
        return w_name

    def print_training_setting(self, settings):
        """ Does what it says. """

        self.logger.info('\nTraining details:')
        for k in settings[self.session_id]:
            if isinstance(settings[self.session_id][k], dict):
                self.logger.info('  {}: '.format(k))
                for kk in settings[self.session_id][k]:
                    self.logger.info(
                        '    {key}: {value}'.format(
                            key=kk, value=str(settings[self.session_id][k][kk])
                        )
                    )
            else:
                self.logger.info(
                    '  {key}: {value}'.format(key=k, value=str(settings[self.session_id][k]))
                )

    def _save_weights(self, epoch, generator, discriminator=None, metric=None, best=False):
        """ Saves the weights of the non-None models. """

        if best:
            gen_path = self.weights_name['generator'].with_name(
                (self.weights_name['generator'].name).format(
                    metric='_best-' + metric, epoch=epoch + 1
                )
            )
        else:
            gen_path = self.weights_name['generator'].with_name(
                (self.weights_name['generator'].name).format(metric='', epoch=epoch + 1)
            )
        # CANT SAVE MODEL DUE TO TF LAYER INSIDE LAMBDA (PIXELSHUFFLE)
        generator.save_weights(gen_path)
        if discriminator:
            if best:
                discr_path = self.weights_name['discriminator'].with_name(
                    (self.weights_name['discriminator'].name).format(
                        metric='_best-' + metric, epoch=epoch + 1
                    )
                )
            else:
                discr_path = self.weights_name['discriminator'].with_name(
                    (self.weights_name['discriminator'].name).format(metric='', epoch=epoch + 1)
                )
            discriminator.model.save_weights(discr_path)
        try:
            self._remove_old_weights(self.max_n_other_weights, max_best=self.max_n_best_weights)
        except Exception as e:
            self.logger.warning('Could not remove weights: {}'.format(e))

    def _remove_old_weights(self, max_n_weights, max_best=5):
        """
        Scans the weights folder and removes all but:
            - the max_best newest 'best' weights.
            - max_n_weights most recent 'others' weights.
        """

        w_list = {}
        w_list['all'] = [w for w in self.callback_paths['weights'].iterdir() if '.hdf5' in w.name]
        w_list['best'] = [w for w in w_list['all'] if 'best' in w.name]
        w_list['others'] = [w for w in w_list['all'] if w not in w_list['best']]
        # remove older best
        epochs_set = {}
        epochs_set['best'] = list(
            set([self.epoch_n_from_weights_name(w.name) for w in w_list['best']])
        )
        epochs_set['others'] = list(
            set([self.epoch_n_from_weights_name(w.name) for w in w_list['others']])
        )
        keep_max = {'best': max_best, 'others': max_n_weights}
        for type in ['others', 'best']:
            if len(epochs_set[type]) > keep_max[type]:
                epoch_list = np.sort(epochs_set[type])[::-1]
                epoch_list = epoch_list[0 : keep_max[type]]
                for w in w_list[type]:
                    if self.epoch_n_from_weights_name(w.name) not in epoch_list:
                        w.unlink()

    def on_epoch_end(self, epoch, losses, generator, discriminator=None, metrics={}):
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
                    self._save_weights(epoch, generator, discriminator, metric=metric, best=True)
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
        Only works if the weights contain 'epoch' followed by 3 integers, for example:
            some-architectureepoch023suffix.hdf5
        """
        try:
            starting_epoch = int(w_name.split('epoch')[1][0:3])
        except Exception as e:
            self.logger.warning(
                'Could not retrieve starting epoch from the weights name: \n{}'.format(w_name)
            )
            self.logger.error(e)
            starting_epoch = 0
        return starting_epoch

    def initialize_training(self, object):
        """Function that is exectured prior to training.

        Wraps up most of the functions of this class:
        load the weights if any are given, generaters names for session and weights,
        creates directories and prints the training session.
        """

        object.weights_generator = self.pretrained_generator_weights
        object.weights_discriminator = self.pretrained_discriminator_weights
        object._load_weights()
        w_name = object.weights_generator
        if w_name:
            last_epoch = self.epoch_n_from_weights_name(w_name.name)
        else:
            last_epoch = 0

        self.callback_paths = self._make_callback_paths()
        self.callback_paths['weights'].mkdir(parents=True)
        self.callback_paths['logs'].mkdir(parents=True)
        object.settings['training_parameters']['starting_epoch'] = last_epoch
        self.settings = self.update_config(object.settings)
        self.print_training_setting(self.settings)
        yaml.dump(
            self.settings, (self.callback_paths['weights'] / self.session_config_name).open('w')
        )
        return last_epoch
