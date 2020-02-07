import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from Text_encoder import Text_encoder
from Generator import Generator
from Discriminator import Discriminator
from Loss import r_precision

VOCAB_SIZE = 13  # TBD after training
EMB_SIZE = 128


class TextGAN(keras.models.Model):
    def __init__(self):
        super(TextGAN, self).__init__()
        self.text_encoder = Text_encoder(VOCAB_SIZE, 64, EMB_SIZE)
        self.generator = Generator((164, ))
        self.discriminator = Discriminator((64, 64))


def fetch_image():
    pass


def fetch_caption():
    pass


if __name__ == '__main__':
    img_batch = fetch_image()
    caption_batch = fetch_caption()

    caption = 'The dog is jumping.'
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(caption)
    print(tokenizer.index_word)
    cap_vector = tokenizer.texts_to_sequences(caption, )
    cap_vector = [token[0] for token in cap_vector if token]
    print(cap_vector)

    caption_emb = Text_encoder(VOCAB_SIZE + 1, 64, EMB_SIZE, drop_out_rate=0.2)(np.array(cap_vector))
    print(caption_emb)
