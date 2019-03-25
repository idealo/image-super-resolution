import numpy as np


def process_array(image_array):
    """ Process a 3-dimensional array into a scaled, 4 dimensional batch of size 1. """

    image_array = image_array / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch


def process_output(output_tensor):
    """ Transforms the 4-dimensional output tensor into a suitable image format. """

    sr_img = output_tensor[0].clip(0, 1) * 255
    sr_img = np.uint8(sr_img)
    return sr_img
