import tensorflow.keras as keras
from tensorflow.keras import layers


class TextEncoder(keras.Model):
    def __init__(self, input_dim, hidden_dim, vector_size, drop_out_rate):
        super(TextEncoder, self).__init__()

        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=vector_size)
        self.drop_out = layers.Dropout(drop_out_rate)
        self.reshape = layers.Reshape((1, -1))
        self.rnn = layers.Bidirectional(layers.LSTM(hidden_dim))

    def call(self, input, **kwargs):
        x = self.embedding(input)
        x = self.drop_out(x)
        x = self.reshape(x)
        x = self.rnn(x)
        return x[-1]






