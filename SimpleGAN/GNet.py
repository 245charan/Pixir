import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Reshape, UpSampling2D, Conv2D, Dense
from tensorflow import keras


class GNet(keras.Model):
    def __init__(self):
        """
        init params
        :param z_input_dim: input dim of z
        """
        super(GNet, self).__init__()
        self.layers_ = []
        self.layers_.append(Dense(512))
        self.layers_.append(LeakyReLU(0.2))
        self.layers_.append(Dense(128 * 16 * 16))
        self.layers_.append(LeakyReLU(0.2))
        self.layers_.append(BatchNormalization())
        self.layers_.append(Reshape((16, 16, 128), input_shape=(128 * 16 * 16,)))
        self.layers_.append(UpSampling2D(size=(2, 2)))
        self.layers_.append(Conv2D(64, (5, 5), padding='same', activation='tanh'))
        self.layers_.append(UpSampling2D(size=(2, 2)))
        self.layers_.append(Conv2D(3, (5, 5), padding='same', activation='tanh'))

    def call(self, input, training=None, mask=None):
        x = input
        for layer in self.layers_:
            x = layer(x)

        return x


if __name__ == '__main__':
    gnet = GNet()
    inputs = tf.random.normal((2, 228))
    output = gnet(inputs)
    print(output.shape)
