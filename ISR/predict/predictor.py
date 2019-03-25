import imageio
import os
import numpy as np
from time import time
from ISR.utils.logger import get_logger
from ISR.utils.image_processing import process_array, process_output


class Predictor:
    """The predictor class handles prediction, given an input model.

    Loads the images in the input directory, executes training given a model
    and saves the results in the output directory.
    Can receive a path for the weights or can let the user browse through the
    weights directory for the desired weights.

    Args:
        input_dir: string, path to the input directory.
        output_dir: string, path to the output directory.
        verbose: bool.

    Attributes:
        extensions: list of accepted image extensions.
        img_ls: list of image files in input_dir.

    Methods:
        get_predictions: given a model and a string containing the weights' path,
            runs the predictions on the images contained in the input directory and
            stores the results in the output directory.
    """

    def __init__(self, input_dir, output_dir='./data/output', verbose=True):

        self.input_dir = input_dir
        self.data_name = os.path.basename(os.path.normpath(self.input_dir))
        self.output_dir = os.path.join(output_dir, self.data_name)
        file_ls = os.listdir(self.input_dir)
        self.logger = get_logger(__name__)
        if not verbose:
            self.logger.setLevel(40)
        self.extensions = ('.jpeg', '.jpg', '.png')  # file extensions that are admitted
        self.img_ls = [file for file in file_ls if file.endswith(self.extensions)]
        if len(self.img_ls) < 1:
            self.logger.error('No valid image files found (check config file).')
            raise ValueError('No valid image files found (check config file).')
        # Create results folder
        if not os.path.exists(self.output_dir):
            self.logger.info('Creating output directory:\n{}'.format(self.output_dir))
            os.makedirs(self.output_dir, exist_ok=True)

    def _load_weights(self):
        """ Invokes the model's load weights function if any weights are provided. """

        if self.weights_path is not None:
            self.logger.info('Loaded weights from \n > {}'.format(self.weights_path))
            # loading by name automatically excludes the vgg layers
            self.model.model.load_weights(self.weights_path)
        else:
            self.logger.error('Error: Weights path not specified (check config file).')
            raise ValueError('Weights path not specified (check config file).')

    def _make_directory_structure(self):
        """ Creates the folder structure from the weights' name. """

        filename = os.path.basename(self.weights_path)
        weights_name, _ = os.path.splitext(filename)
        subdirs = weights_name.split('_')
        self.basepath = os.path.join(*subdirs)

    def get_predictions(self, model, weights_path):
        """ Runs the prediction. """

        self.model = model
        self.weights_path = weights_path
        self._load_weights()
        self._make_directory_structure()
        out_folder = os.path.join(self.output_dir, self.basepath)
        self.logger.info('Results in:\n > {}'.format(out_folder))
        if os.path.exists(out_folder):
            self.logger.warning('Directory exists, might overwrite files')
        os.makedirs(out_folder, exist_ok=True)
        # Predict and store
        for img_name in self.img_ls:
            input_path = os.path.join(self.input_dir, img_name)
            output_path = os.path.join(out_folder, img_name)
            self.logger.info('Processing file\n > {}'.format(input_path))
            start = time()
            sr_img = self._forward_pass(input_path)
            end = time()
            self.logger.info('Elapsed time: {}s'.format(end - start))
            self.logger.info('Result in: {}'.format(output_path))
            imageio.imwrite(output_path, sr_img)

    def _forward_pass(self, file_path):
        lr_img = imageio.imread(file_path)
        if lr_img.shape[2] == 3:
            lr_img = process_array(lr_img)
            sr_img = self.model.model.predict(lr_img)
            sr_img = process_output(sr_img)
            return sr_img
        else:
            self.logger.error('{} is not an image with 3 channels.'.format(file_path))
