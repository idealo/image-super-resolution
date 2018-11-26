import os
from utils.generator import Generator
from logging import info, error, debug, warning
from utils.utils import browse_weights, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint

class Trainer:
    """Trainer class.
    Given an input model, handles the various folders defined in config.json,
    creates appropriate generators, tensorboard instance and checkpoints.

    Initial weights can be passed as input or can be iteratively selected on the command line.
    By default, the weights are randomly initialized.
    """
    def __init__(self, train_arguments):
        if (train_arguments['select_weights'] is not 0) or (train_arguments['weights_path'] is not ''):
            self.weights_path, self.last_epoch = browse_weights(weights_path=train_arguments['weights_path'])
        else:
            self.weights_path = ''
            self.last_epoch = 0

        train_folders = {'HR': train_arguments['training_labels'],
                         'LR': train_arguments['training_input']}
        valid_folders = {'HR': train_arguments['validation_labels'],
                         'LR': train_arguments['validation_input']}
        self.folders = {'train': train_folders,
                        'valid': valid_folders}

        # self.model = load_model(model_parameters, train_arguments.add_vgg)
        self.steps_per_epoch = train_arguments['steps_per_epoch']
        self.n_validation_samples = train_arguments['n_validation_samples']
        self.weights_dir = train_arguments['weights_dir']
        self.batch_size = train_arguments['batch_size']
        self.patch_size = train_arguments['patch_size']
        self.dataname = train_arguments['data_name']
        self.verbose = train_arguments['verbose']
        self.epochs = train_arguments['epochs']
        self.log_dir = train_arguments['log_dir']

    def get_generator(self, type):
        return Generator(input_folder=self.folders[type]['LR'],
                         label_folder=self.folders[type]['HR'],
                         batch_size=self.batch_size,
                         patch_size=self.patch_size,
                         scale=self.scale,
                         n_validation_samples=self.n_validation_samples,
                         mode=type)

    def make_generators(self):
        self.generators = {}
        for type in ['train', 'valid']:
            self.generators[type] = self.get_generator(type)

    def load_weights(self, model):
        if self.weights_path is not '':
            model.rdn.load_weights(self.weights_path)
            info('>>> Loaded weights from')
            info(self.weights_path)
        if self.last_epoch > 0:
            info('>>> Starting from epoch', self.last_epoch + 1, '\n')

    def make_checkpoints_paths(self):
        self.checkpoint_paths = {}
        for metric in ['loss', 'PSNR']:
            weights_name = '_'.join([''.join(['weights', self.dataname]),
                                     'E{epoch:03d}',
                                     'X{}'.format(str(self.scale)),
                                     '{}.hdf5'.format(metric)])
            self.checkpoint_paths[metric] = os.path.join(self.weights_dir, weights_name)

    def train_model(self, model):
        self.load_weights(model)
        self.scale = model.scale
        self.make_generators()
        self.make_checkpoints_paths()

        checkpoint_loss = ModelCheckpoint(self.checkpoint_paths['loss'],
                                          save_weights_only=True,
                                          verbose=self.verbose,
                                          save_best_only=True,
                                          monitor='loss',
                                          mode='min')

        checkpoint_PSNR = ModelCheckpoint(self.checkpoint_paths['PSNR'],
                                          save_weights_only=True,
                                          verbose=self.verbose,
                                          save_best_only=True,
                                          monitor='PSNR',
                                          mode='max')

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        tboard = TensorBoard(self.log_dir)
        callbacks_list = [checkpoint_loss, checkpoint_PSNR, tboard]

        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)

        info('>>> Starting training.')
        model.rdn.fit_generator(steps_per_epoch=self.steps_per_epoch,
                                     validation_data=self.generators['valid'],
                                     initial_epoch=self.last_epoch,
                                     generator=self.generators['train'],
                                     use_multiprocessing=True,
                                     callbacks=callbacks_list,
                                     verbose=self.verbose,
                                     epochs=self.epochs)
