import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from TextEncoder import TextEncoder
from GNet import GNet
from DNet import DNet
from Read_data import *
from resize_image import resize_image
from Text_Preprocessing import text_preprocessing
from Train import train
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

VOCAB_SIZE = 14  # TBD after training
HIDDEN_DIM = 64
EMB_SIZE = 128
DROPOUT_RATE = 0.3


class SimpleGAN(keras.models.Model):
    def __init__(self, vocab_size):
        super(SimpleGAN, self).__init__()
        self.text_encoder = TextEncoder(vocab_size, HIDDEN_DIM, EMB_SIZE, dropout_rate=DROPOUT_RATE)
        self.generator = GNet()
        self.discriminator = DNet(dropout_rate=DROPOUT_RATE)

    def call(self, input, fake_captions, **kwargs):
        """

        :param input: tf.data.Dataset(images, texts)
        :param fake_captions: 가짜 캡션 vector
        :return: (real_text, fake_image) 예측값, (real_text, real_image) 예측값, (fake_text, real_image) 예측값
        """
        emb_vectors = self.text_encoder(input[1])
        fake_emb_vectors = self.text_encoder(fake_captions)
        noise = tf.random.normal((emb_vectors.shape[0], 100))
        generator_input = tf.concat([emb_vectors, noise], -1)
        fake_images = self.generator(generator_input)
        fake_image_pred = self.discriminator(fake_images, emb_vectors)
        real_image_pred = self.discriminator(input[0], emb_vectors)
        fake_caption_pred = self.discriminator(input[0], fake_emb_vectors)

        return fake_image_pred, real_image_pred, fake_caption_pred

    def generate(self, text_sequence):
        emb_vector = self.text_encoder(tf.reshape(text_sequence, [1, -1]))
        noise = tf.random.normal((1, 100))
        generator_input = tf.concat([emb_vector, noise], -1)
        fake_image = self.generator(generator_input)
        return fake_image


if __name__ == '__main__':
    # model = keras.Sequential([
    #     keras.layers.Dense(32, input_shape=(4,))
    # ])
    # print(model.trainable_variables)
    # model = SimpleGAN()
    # ls = [1, 2, 3, 4]
    # print(derangement(ls))
    # img_batch = fetch_image()
    # caption_batch = fetch_caption()
    #
    # caption = ['The dog is jumping', 'the cat is flying']
    # tokenizer = keras.preprocessing.text.Tokenizer()
    # tokenizer.fit_on_texts(caption)
    # print(tokenizer.index_word)
    # cap_vector = np.array(tokenizer.texts_to_sequences(caption))
    # # cap_vector = [token[0] for token in cap_vector if token]
    # print(cap_vector)
    #
    # images = np.random.random((2, 64, 64, 3))
    #
    # predictions = model([images, cap_vector])
    # print(len(model.trainable_variables))
    # print(predictions)

    # caption_emb = TextEncoder(VOCAB_SIZE, EMB_SIZE, 64, dropout_rate=0.2)(np.array(cap_vector))
    # print(caption_emb.shape)
    #
    # noise = tf.random.normal((caption_emb.shape[0], 100))
    # print(noise.shape)
    #
    # gan_vector = tf.concat([caption_emb, noise], -1)
    # print(gan_vector.shape)
    #
    # generated_img = GNet()(gan_vector)
    # print(generated_img.shape)
    #
    # img = generated_img[0]
    # print(img.shape)
    #
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    #
    # prediction = DNet()(generated_img, caption_emb)
    # print(prediction)

    texts = read_labels(1)[:100]

    image_resize = resize_image(64, 64)
    texts_vectors, vocab_size = text_preprocessing(texts)

    model = SimpleGAN(vocab_size)
    train(texts_vectors, image_resize, model, epochs=50, batch_size=16)
    model.save_weights('SimpleGAN.h5')

    print(texts_vectors[0])
    img = model.generate(texts_vectors[0])
    img = (img + 1) * 255
    plt.imshow(img.numpy().squeeze())
    plt.axis('off')
    plt.show()

    # image_resize = image_resize / 255 - 1
    # pred = model([image_resize[:10], texts_vectors[:10]], texts_vectors[10:20])
    # print(len(model.text_encoder.trainable_variables + model.generator.trainable_variables))
    # print()
    # # print(len(model.generator.trainable_variables))
    # print()
    # print(len(model.discriminator.trainable_variables))
