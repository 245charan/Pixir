import tensorflow.keras as keras
from tensorflow.keras import layers


class TextEncoder(keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_dim, dropout_rate=0.2):
        super(TextEncoder, self).__init__()

        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True)
        self.drop_out = layers.Dropout(dropout_rate)
        # self.reshape = layers.Reshape((1, -1))
        self.rnn = layers.Bidirectional(layers.LSTM(hidden_dim))

    def call(self, input, **kwargs):
        x = self.embedding(input)
        x = self.drop_out(x)
        # x = self.reshape(x)
        x = self.rnn(x)
        return x






