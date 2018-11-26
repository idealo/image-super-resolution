import os
import numpy as np
from time import time
from imageio import imwrite, imread
from logging import info, error, debug, warning
from utils.utils import browse_weights, load_model


class Predictor:
    """The predictor class handles prediction, given an input model.
    Reads input files from the folder specified in config.json.
    Saves results in output folder specified in config.json.

    Can receive a path for the weights or can let the user browse through the
    weights directory for the desired weights.
    """
    def __init__(self, test_arguments):
        # Select pre-trained weights
        self.weights_path, _ = browse_weights(weights_path=test_arguments['weights_path'])
        self.results_folder = test_arguments['results_folder']
        self.test_folder = test_arguments['test_folder']
        self.verbose = test_arguments['verbose']
        # Create results folder
        if not os.path.exists(self.results_folder):
            info('Creating directory for results:\n', self.results_folder)
            os.mkdir(self.results_folder)

    def load_weights(self):
        if self.weights_path is not None:
            self.model.rdn.load_weights(self.weights_path)
            info('>>> Loaded weights from')
            info(self.weights_path)
        else:
            error('>>> ERROR: LOAD VALID WEIGHTS')
            raise

    def get_predictions(self, model):
        self.model = model
        self.load_weights()
        # Predict and store
        for file in os.listdir(self.test_folder):
            file_path = os.path.join(self.test_folder, file)
            output_path = os.path.join(self.results_folder, file)
            info('\n>>> Processing', file_path)
            start = time()
            SR_img = self.forward_pass(file_path)[0]
            end = time()
            info('Elapsed time', end - start)
            info('Result in', output_path)
            imwrite(output_path, SR_img)

    def forward_pass(self, file_path):
        LR_img = imread(str(file_path)) / 255.
        LR_img = np.expand_dims(LR_img, axis=0)
        SR_img = self.model.rdn.predict(LR_img)
        return np.clip(SR_img, 0, 1) * 255
