import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential, layers


class text_encoder(keras.Model):
    def __init__(self, input_dim, hidden_dim, vector_size, drop_out_rate):
        super(text_encoder, self).__init__()

        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=vector_size)
        self.drop_out = layers.Dropout(drop_out_rate)
        self.rnn = layers.Bidirectional(layers.LSTM(hidden_dim))

    def call(self, input, **kwargs):
        x = self.embedding(input)
        x = self.drop_out(x)
        x = self.rnn(x)
        return x




