from keras.layers import LeakyReLU, BatchNormalization, Reshape, UpSampling2D, Conv2D
from keras.optimizers import Adam
from tensorflow.keras.models import Model
########## G networks ###########
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Dense


class GAN:
    def __init__(self, learning_rate, z_input_dim):
        """
        init params

        :param learning_rate: learning rate of optimizer
        :param z_input_dim: input dim of z
        """
        super(GAN, self).__init__()
        self.Dense1 = Dense(512, input_dim=z_input_dim)
        self.LeakyReLU1 = LeakyReLU(0.2)
        self.Dense2 = Dense(128 * 7 * 7)
        self.LeakyReLU2 = LeakyReLU(0.2)
        self.BatchNormalization2 = BatchNormalization()
        self.Reshape3 = Reshape((128, 7, 7), input_shape=(128 * 7 * 7,))
        self.UpSampling2D3 = UpSampling2D(size=(2, 2))
        self.Conv2D3 = Conv2D(64, (5, 5), padding='same', activation='tanh')
        self.UpSampling2D3 = UpSampling2D(size=(2, 2))
        self.Conv2D4 = Conv2D(1, (5, 5), padding='same', activation='tanh')

    def call(self, input):
        x = self.Dense1(input)
        x = self.LeakyReLU1(x)
        x = self.Dense2(x)
        x = self.LeakyReLU2(x)
        x = self.BatchNormalization2(x)
        x = self.Reshape3(x)
        x = self.UpSampling2D3(x)
        x = self.Conv2D3(x)
        x = self.UpSampling2D3(x)
        x = self.Conv2D4(x)

        return x
