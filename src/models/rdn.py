from utils.metrics import PSNR
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import UpSampling2D, concatenate
from keras.layers import Input, Activation, Add, Conv2D


def make_model(**model_parameters):
    """Returns the model.
    Used to select the model.
    """
    return RDN(**model_parameters)


class RDN:
    def __init__(self, D, C, G, G0=64, c_dim=3, scale=2, kernel_size=3, learning_rate=1e-5):
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        self.c_dim = c_dim
        self.scale = scale
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate

        optimizer_rdn = Adam(lr=self.learning_rate)

        self.rdn = self.build_rdn(optimizer_rdn)

    def UPN(self, input_layer):
        """Upscaling layers."""
        x = Conv2D(64, kernel_size=5, strides=1, padding='same', name='UPN1')(input_layer)
        x = Activation('relu', name='UPN1_Relu')(x)
        x = Conv2D(32, kernel_size=3, padding='same', name='UPN2')(x)
        x = Activation('relu', name='UPN2_Relu')(x)
        x = Conv2D(self.c_dim * self.scale ** 2, kernel_size=3, padding='same', name='UPN3')(x)
        x = UpSampling2D(size=self.scale, name='UPsample')(x)
        return x

    def RDBs(self, input_layer):
        """RDBs blocks.
        Input F_0, output concatenation of RDBs output feature maps.
        # output G0 feature maps
        """
        rdb_concat = list()
        rdb_in = input_layer
        for d in range(1, self.D + 1):
            x = rdb_in
            for c in range(1, self.C + 1):
                F_dc = Conv2D(self.G, kernel_size=self.kernel_size, padding='same', name='F_%d_%d' % (d, c))(x)
                F_dc = Activation('relu', name='F_%d_%d_Relu' % (d, c))(F_dc)
                # concatenate input and output of ConvRelu block
                # x = [input_layer,F_11(input_layer),F_12([input_layer,F_11(input_layer)]), F_13..]
                x = concatenate([x, F_dc], axis=3, name='RDB_Concat_%d_%d' % (d, c))
            # 1x1 convolution (Local Feature Fusion)
            x = Conv2D(self.G0, kernel_size=1, name='LFF_%d' % (d))(x)
            # Local Residual Learning F_{i,LF} + F_{i-1}
            rdb_in = Add(name='LRL_%d' % (d))([x, rdb_in])
            rdb_concat.append(rdb_in)

        assert len(rdb_concat) == self.D

        return concatenate(rdb_concat, axis=3, name='LRLs_Concat')

    def build_rdn(self, optimizer):
        LR_input = Input(shape=(None, None, 3), name='LR')
        F_m1 = Conv2D(self.G0, kernel_size=self.kernel_size, padding='same', name='F_m1')(LR_input)
        F_0 = Conv2D(self.G0, kernel_size=self.kernel_size, padding='same', name='F_0')(F_m1)
        FD = self.RDBs(F_0)
        # Global Feature Fusion
        # 1x1 Conv of concat RDB layers -> G0 feature maps
        GFF1 = Conv2D(self.G0, kernel_size=1, padding='same', name='GFF_1')(FD)
        GFF2 = Conv2D(self.G0, kernel_size=self.kernel_size, padding='same', name='GFF_2')(GFF1)
        # Global Residual Learning for Dense Features
        FDF = Add(name='FDF')([GFF2, F_m1])
        # Upscaling
        FU = self.UPN(FDF)
        # Compose SR image
        SR = Conv2D(self.c_dim, kernel_size=self.kernel_size, padding='same', name='SR')(FU)

        # Single component loss
        model = Model(inputs=LR_input, outputs=SR)

        model.compile(loss='mse', optimizer=optimizer, metrics=[PSNR])
        return model
