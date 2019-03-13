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
