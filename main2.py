import tensorflow as tf

from test.data import read_labels, resize_image, text_preprocessing
from test.models import build_bert_encoder, build_generator, build_discriminator
from test.train import train

if __name__ == '__main__':
    BATCH_SIZE = 16
    NUM_EPOCHS = 50

    image_resize = resize_image(64, 64)

    texts = read_labels(1)[:100]
    tokens, masks, segments = text_preprocessing(texts)

    bert_model = build_bert_encoder()
    text_vectors = bert_model.predict([tokens, masks, segments])

    generator = build_generator((32, ))
    discriminator = build_discriminator((64, 64, 3), (1024, ))

    seeds = text_vectors[:16]

    dataset = tf.data.Dataset.from_tensor_slices((image_resize, text_vectors))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(BATCH_SIZE)

    train(dataset, generator, discriminator, seeds, epochs=50)
