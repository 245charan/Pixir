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
        self.learning_rate = learning_rate
        self.z_input_dim = z_input_dim
        self.D = self.discriminator()
        self.G = self.generator()
        self.GD = self.combined()

    """
    Generator
    """
    G = Sequential()
    G.add(Dense(512, input_dim=self.z_input_dim))
    G.add(LeakyReLU(0.2))
    G.add(Dense(128 * 7 * 7))
    G.add(LeakyReLU(0.2))
    G.add(BatchNormalization())
    G.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
    G.add(UpSampling2D(size=(2, 2)))
    G.add(Conv2D(64, (5, 5), padding='same', activation='tanh'))
    G.add(UpSampling2D(size=(2, 2)))
    G.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))

    adam = Adam(lr=self.learning_rate, beta_1=0.5)
    G.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return G
