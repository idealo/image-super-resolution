import yaml
import numpy as np
from time import time
from tqdm import tqdm
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K
from ISR.utils.datahandler import DataHandler
from ISR.utils.train_helper import TrainerHelper
from ISR.utils.metrics import PSNR
from ISR.utils.metrics import PSNR_Y
from ISR.utils.logger import get_logger
from ISR.utils.utils import check_parameter_keys


class Trainer:
    """Class object to setup and carry the training.

    Takes as input a generator that produces SR images.
    Conditionally, also a discriminator network and a feature extractor
        to build the components of the perceptual loss.
    Compiles the model(s) and trains in a GANS fashion if a discriminator is provided, otherwise
    carries a regular ISR training.

    Args:
        generator: Keras model, the super-scaling, or generator, network.
        discriminator: Keras model, the discriminator network for the adversarial
            component of the perceptual loss.
        feature_extractor: Keras model, feature extractor network for the deep features
            component of perceptual loss function.
        lr_train_dir: path to the directory containing the Low-Res images for training.
        hr_train_dir: path to the directory containing the High-Res images for training.
        lr_valid_dir: path to the directory containing the Low-Res images for validation.
        hr_valid_dir: path to the directory containing the High-Res images for validation.
        learning_rate: float.
        loss_weights: dictionary, use to weigh the components of the loss function.
            Contains 'generator' for the generator loss component, and can contain 'discriminator' and 'feature_extractor'
            for the discriminator and deep features components respectively.
        logs_dir: path to the directory where the tensorboard logs are saved.
        weights_dir: path to the directory where the weights are saved.
        dataname: string, used to identify what dataset is used for the training session.
        weights_generator: path to the pre-trained generator's weights, for transfer learning.
        weights_discriminator: path to the pre-trained discriminator's weights, for transfer learning.
        n_validation:integer, number of validation samples used at training from the validation set.
        flatness: dictionary. Determines determines the 'flatness' threshold level for the training patches.
            See the TrainerHelper class for more details.
        lr_decay_frequency: integer, every how many epochs the learning rate is reduced.
        lr_decay_factor: 0 < float <1, learning rate reduction multiplicative factor.

    Methods:
        train: combines the networks and triggers training with the specified settings.

    """

    def __init__(
        self,
        generator,
        discriminator,
        feature_extractor,
        lr_train_dir,
        hr_train_dir,
        lr_valid_dir,
        hr_valid_dir,
        loss_weights={'generator': 1.0, 'discriminator': 0.003, 'feature_extractor': 1 / 12},
        log_dirs={'logs': 'logs', 'weights': 'weights'},
        fallback_save_every_n_epochs=2,
        dataname=None,
        weights_generator=None,
        weights_discriminator=None,
        n_validation=None,
        flatness={'min': 0.0, 'increase_frequency': None, 'increase': 0.0, 'max': 0.0},
        learning_rate={'initial_value': 0.0004, 'decay_frequency': 100, 'decay_factor': 0.5},
        adam_optimizer={'beta1': 0.9, 'beta2': 0.999, 'epsilon': None},
        losses={
            'generator': 'mae',
            'discriminator': 'binary_crossentropy',
            'feature_extractor': 'mse',
        },
        metrics={'generator': 'PSNR_Y'},
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.scale = generator.scale
        self.lr_patch_size = generator.patch_size
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.weights_generator = weights_generator
        self.weights_discriminator = weights_discriminator
        self.adam_optimizer = adam_optimizer
        self.dataname = dataname
        self.flatness = flatness
        self.n_validation = n_validation
        self.losses = losses
        self.log_dirs = log_dirs
        self.metrics = metrics
        if self.metrics['generator'] == 'PSNR_Y':
            self.metrics['generator'] = PSNR_Y
        elif self.metrics['generator'] == 'PSNR':
            self.metrics['generator'] = PSNR
        self._parameters_sanity_check()
        self.model = self._combine_networks()

        self.settings = {}
        self.settings['training_parameters'] = locals()
        self.settings['training_parameters']['lr_patch_size'] = self.lr_patch_size
        self.settings = self.update_training_config(self.settings)

        self.logger = get_logger(__name__)

        self.helper = TrainerHelper(
            generator=self.generator,
            weights_dir=log_dirs['weights'],
            logs_dir=log_dirs['logs'],
            lr_train_dir=lr_train_dir,
            feature_extractor=self.feature_extractor,
            discriminator=self.discriminator,
            dataname=dataname,
            weights_generator=self.weights_generator,
            weights_discriminator=self.weights_discriminator,
            fallback_save_every_n_epochs=fallback_save_every_n_epochs,
        )

        self.train_dh = DataHandler(
            lr_dir=lr_train_dir,
            hr_dir=hr_train_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=None,
        )
        self.valid_dh = DataHandler(
            lr_dir=lr_valid_dir,
            hr_dir=hr_valid_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=n_validation,
        )

    def _parameters_sanity_check(self):
        """ Parameteres sanity check. """

        if self.discriminator:
            assert self.lr_patch_size * self.scale == self.discriminator.patch_size
            self.adam_optimizer
        if self.feature_extractor:
            assert self.lr_patch_size * self.scale == self.feature_extractor.patch_size

        check_parameter_keys(
            self.learning_rate,
            needed_keys=['initial_value'],
            optional_keys=['decay_factor', 'decay_frequency'],
            default_value=None,
        )
        check_parameter_keys(
            self.flatness,
            needed_keys=[],
            optional_keys=['min', 'increase_frequency', 'increase', 'max'],
            default_value=0.0,
        )
        check_parameter_keys(
            self.adam_optimizer,
            needed_keys=['beta1', 'beta2'],
            optional_keys=['epsilon'],
            default_value=None,
        )
        check_parameter_keys(self.log_dirs, needed_keys=['logs', 'weights'])

    def _combine_networks(self):
        """
        Constructs the combined model which contains the generator network,
        as well as discriminator and geature extractor, if any are defined.
        """

        lr = Input(shape=(self.lr_patch_size,) * 2 + (3,))
        sr = self.generator.model(lr)
        outputs = [sr]
        losses = [self.losses['generator']]
        loss_weights = [self.loss_weights['generator']]

        if self.discriminator:
            self.discriminator.model.trainable = False
            validity = self.discriminator.model(sr)
            outputs.append(validity)
            losses.append(self.losses['discriminator'])
            loss_weights.append(self.loss_weights['discriminator'])
        if self.feature_extractor:
            self.feature_extractor.model.trainable = False
            sr_feats = self.feature_extractor.model(sr)
            outputs.extend([*sr_feats])
            losses.extend([self.losses['feature_extractor']] * len(sr_feats))
            loss_weights.extend(
                [self.loss_weights['feature_extractor'] / len(sr_feats)] * len(sr_feats)
            )
        combined = Model(inputs=lr, outputs=outputs)
        # https://stackoverflow.com/questions/42327543/adam-optimizer-goes-haywire-after-200k-batches-training-loss-grows
        optimizer = Adam(
            beta_1=self.adam_optimizer['beta1'],
            beta_2=self.adam_optimizer['beta2'],
            lr=self.learning_rate['initial_value'],
            epsilon=self.adam_optimizer['epsilon'],
        )
        combined.compile(
            loss=losses, loss_weights=loss_weights, optimizer=optimizer, metrics=self.metrics
        )
        return combined

    def _lr_scheduler(self, epoch):
        """ Scheduler for the learning rate updates. """

        n_decays = epoch // self.learning_rate['decay_frequency']
        lr = self.learning_rate['initial_value'] * (self.learning_rate['decay_factor'] ** n_decays)
        # no lr below minimum control 10e-7
        return max(1e-7, lr)

    def _flatness_scheduler(self, epoch):
        if self.flatness['increase']:
            n_increases = epoch // self.flatness['increase_frequency']
        else:
            return self.flatness['min']

        f = self.flatness['min'] + n_increases * self.flatness['increase']

        return min(self.flatness['max'], f)

    def _load_weights(self):
        """
        Loads the pretrained weights from the given path, if any is provided.
        If a discriminator is defined, does the same.
        """

        if self.weights_generator:
            self.model.get_layer('generator').load_weights(self.weights_generator)

        if self.discriminator:
            if self.weights_discriminator:
                self.model.get_layer('discriminator').load_weights(self.weights_discriminator)
                self.discriminator.model.load_weights(self.weights_discriminator)

    def _format_losses(self, prefix, losses, model_metrics):
        """ Creates a dictionary for tensorboard tracking. """

        return dict(zip([prefix + m for m in model_metrics], losses))

    def update_training_config(self, settings):
        """ Summarizes training setting. """

        _ = settings['training_parameters'].pop('weights_generator')
        _ = settings['training_parameters'].pop('self')
        _ = settings['training_parameters'].pop('generator')
        _ = settings['training_parameters'].pop('discriminator')
        _ = settings['training_parameters'].pop('feature_extractor')
        settings['generator'] = {}
        settings['generator']['name'] = self.generator.name
        settings['generator']['parameters'] = self.generator.params
        settings['generator']['weights_generator'] = self.weights_generator

        _ = settings['training_parameters'].pop('weights_discriminator')
        if self.discriminator:
            settings['discriminator'] = {}
            settings['discriminator']['name'] = self.discriminator.name
            settings['discriminator']['weights_discriminator'] = self.weights_discriminator
        else:
            settings['discriminator'] = None

        if self.discriminator:
            settings['feature_extractor'] = {}
            settings['feature_extractor']['name'] = self.feature_extractor.name
            settings['feature_extractor']['layers'] = self.feature_extractor.layers_to_extract
        else:
            settings['feature_extractor'] = None

        return settings

    def train(self, epochs, steps_per_epoch, batch_size, monitored_metrics):
        """
        Carries on the training for the given number of epochs.
        Sends the losses to Tensorboard.

        Args:
            epochs: how many epochs to train for.
            steps_per_epoch: how many batches epoch.
            batch_size: amount of images per batch.
            monitored_metrics: dictionary, the keys are the metrics that are monitored for the weights
                saving logic. The values are the mode that trigger the weights saving ('min' vs 'max').
        """

        self.settings['training_parameters']['steps_per_epoch'] = steps_per_epoch
        self.settings['training_parameters']['batch_size'] = batch_size
        starting_epoch = self.helper.initialize_training(
            self
        )  # load_weights, creates folders, creates basename

        self.tensorboard = TensorBoard(log_dir=self.helper.callback_paths['logs'])
        self.tensorboard.set_model(self.model)

        # validation data
        validation_set = self.valid_dh.get_validation_set(batch_size)
        y_validation = [validation_set['hr']]
        if self.discriminator:
            discr_out_shape = list(self.discriminator.model.outputs[0].shape)[1:4]
            valid = np.ones([batch_size] + discr_out_shape)
            fake = np.zeros([batch_size] + discr_out_shape)
            validation_valid = np.ones([len(validation_set['hr'])] + discr_out_shape)
            y_validation.append(validation_valid)
        if self.feature_extractor:
            validation_feats = self.feature_extractor.model.predict(validation_set['hr'])
            y_validation.extend([*validation_feats])

        for epoch in range(starting_epoch, epochs):
            self.logger.info('Epoch {e}/{tot_eps}'.format(e=epoch, tot_eps=epochs))
            K.set_value(self.model.optimizer.lr, self._lr_scheduler(epoch=epoch))
            self.logger.info('Current learning rate: {}'.format(K.eval(self.model.optimizer.lr)))

            flatness = self._flatness_scheduler(epoch)
            if flatness:
                self.logger.info('Current flatness treshold: {}'.format(flatness))

            epoch_start = time()
            for step in tqdm(range(steps_per_epoch)):
                batch = self.train_dh.get_batch(batch_size, flatness=flatness)
                y_train = [batch['hr']]
                training_losses = {}

                ## Discriminator training
                if self.discriminator:
                    sr = self.generator.model.predict(batch['lr'])
                    d_loss_real = self.discriminator.model.train_on_batch(batch['hr'], valid)
                    d_loss_fake = self.discriminator.model.train_on_batch(sr, fake)
                    d_loss_fake = self._format_losses(
                        'train_d_fake_', d_loss_fake, self.discriminator.model.metrics_names
                    )
                    d_loss_real = self._format_losses(
                        'train_d_real_', d_loss_real, self.discriminator.model.metrics_names
                    )
                    training_losses.update(d_loss_real)
                    training_losses.update(d_loss_fake)
                    y_train.append(valid)

                ## Generator training
                if self.feature_extractor:
                    hr_feats = self.feature_extractor.model.predict(batch['hr'])
                    y_train.extend([*hr_feats])

                model_losses = self.model.train_on_batch(batch['lr'], y_train)
                model_losses = self._format_losses('train_', model_losses, self.model.metrics_names)
                training_losses.update(model_losses)

                self.tensorboard.on_epoch_end(epoch * steps_per_epoch + step, training_losses)
                self.logger.debug('Losses at step {s}:\n {l}'.format(s=step, l=training_losses))

            elapsed_time = time() - epoch_start
            self.logger.info('Epoch {} took {:10.1f}s'.format(epoch, elapsed_time))

            validation_losses = self.model.evaluate(
                validation_set['lr'], y_validation, batch_size=batch_size
            )
            validation_losses = self._format_losses(
                'val_', validation_losses, self.model.metrics_names
            )

            if epoch == starting_epoch:
                remove_metrics = []
                for metric in monitored_metrics:
                    if (metric not in training_losses) and (metric not in validation_losses):
                        msg = ' '.join([metric, 'is NOT among the model metrics, removing it.'])
                        self.logger.error(msg)
                        remove_metrics.append(metric)
                for metric in remove_metrics:
                    _ = monitored_metrics.pop(metric)

            # should average train metrics
            end_losses = {}
            end_losses.update(validation_losses)
            end_losses.update(training_losses)

            self.helper.on_epoch_end(
                epoch=epoch,
                losses=end_losses,
                generator=self.model.get_layer('generator'),
                discriminator=self.discriminator,
                metrics=monitored_metrics,
            )
            self.tensorboard.on_epoch_end(epoch, validation_losses)
        self.tensorboard.on_train_end(None)
