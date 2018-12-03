import keras.backend as K


def PSNR(y_true, y_pred, MAXp=1):
    """
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    """
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
