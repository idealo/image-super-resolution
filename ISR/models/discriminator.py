from keras.layers import concatenate, Flatten, Input, Activation, Dense, Conv2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam


class Discriminator:
    """
    Implementation of the discriminator network for the adversarial
    component of the perceptual loss.

    Args:
        patch_size: integer, determines input size as (patch_size, patch_size, 3).
        kernel_size: size of the kernel in the conv blocks.

    Attributes:
        model: Keras model.
        name: name used to identify what discriminator is used during GANs training.
        model.name: identifies this network as the discriminator network
            in the compound model built by the trainer class.
        block_param: dictionary, determines the number of filters and the strides for each
            conv block.

    """

    def __init__(self, patch_size, kernel_size=3):
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.block_param = {}
        self.block_param['filters'] = (64, 128, 128, 256, 256, 512, 512)
        self.block_param['strides'] = (2, 1, 2, 1, 1, 1, 1)
        self.block_num = len(self.block_param['filters'])
        self.model = self._build_disciminator()
        optimizer = Adam(0.0002, 0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.name = 'discriminator'
        self.name = 'srgan-large'

    def _conv_block(self, input, filters, strides, batch_norm=True, count=None):
        """ Convolutional layer + Leaky ReLU + conditional BN. """

        x = Conv2D(
            filters,
            kernel_size=self.kernel_size,
            strides=strides,
            padding='same',
            name='Conv_{}'.format(count),
        )(input)
        x = LeakyReLU(alpha=0.2)(x)
        if batch_norm:
            x = BatchNormalization(momentum=0.8)(x)
        return x

    def _build_disciminator(self):
        """ Puts the discriminator's layers together. """

        HR = Input(shape=(self.patch_size, self.patch_size, 3))
        x = self._conv_block(HR, filters=64, strides=1, batch_norm=False, count=1)
        for i in range(self.block_num):
            x = self._conv_block(
                x,
                filters=self.block_param['filters'][i],
                strides=self.block_param['strides'][i],
                count=i + 2,
            )
        x = Dense(self.block_param['filters'][-1] * 2, name='Dense_1024')(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = Flatten()(x)
        x = Dense(1, name='Dense_last')(x)
        HR_v_SR = Activation('sigmoid')(x)

        discriminator = Model(inputs=HR, outputs=HR_v_SR)
        return discriminator
