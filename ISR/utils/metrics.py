import keras.backend as K


def PSNR(y_true, y_pred, MAXp=1):
    """
    Evaluates the PSNR value:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    Args:
        y_true: ground truth.
        y_pred: predicted value.
        MAXp: maximum value of the pixel range (default=1).
    """
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


def RGB_to_Y(image):
    """ Image has values from 0 to 1. """

    R = image[:, :, :, 0]
    G = image[:, :, :, 1]
    B = image[:, :, :, 2]

    Y = 16 + (65.738 * R) + 129.057 * G + 25.064 * B
    return Y / 255.0


def PSNR_Y(y_true, y_pred, MAXp=1):
    """
    Evaluates the PSNR value on the Y channel:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    Args:
        y_true: ground truth.
        y_pred: predicted value.
        MAXp: maximum value of the pixel range (default=1).
    """
    y_true = RGB_to_Y(y_true)
    y_pred = RGB_to_Y(y_pred)
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
