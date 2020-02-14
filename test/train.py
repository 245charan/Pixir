import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
import matplotlib.pyplot as plt

noise_dim = 100
batch_size = 32


def generator_loss(fake_image_output):
    return binary_crossentropy(tf.ones_like(fake_image_output), fake_image_output, from_logits=True)


def discriminator_loss(real_output, fake_image_output, fake_text_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
    fake_image_output = binary_crossentropy(tf.zeros_like(fake_image_output), fake_image_output, from_logits=True)
    fake_text_output = binary_crossentropy(tf.zeros_like(fake_text_output), fake_text_output, from_logits=True)
    total_loss = real_loss + fake_image_output + fake_text_output
    return total_loss


def train(dataset, generator, discriminator, seeds, epochs, gen_lr=1e-4, disc_lr=1e-4):
    g_optimizer = Adam(gen_lr)
    d_optimizer = Adam(disc_lr)

    for epoch in range(epochs):
        start = time.time()
        for data in dataset:
            images = data[0]
            texts = data[1]
            fake_texts = derangement(texts)
            train_step(images, texts, fake_texts, generator, g_optimizer, discriminator, d_optimizer)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            plot_generated_images(epoch, generator, seeds, random_dim=100, dim=(4, 4))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def derangement(list):
    while True:
        shuffled_list = tf.random.shuffle(list)
        for i in range(len(list)):
            if all(tf.equal(list[i], shuffled_list[i])):
                break
        else:
            break
    return shuffled_list


def plot_generated_images(epoch, generator, texts, random_dim=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[texts.shape[0], random_dim])
    inputs = tf.concat([noise, texts], axis=-1)
    generated_images = generator.predict(inputs)
    generated_images = (generated_images + 1) * 127.5

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

    # if e == 1 or e % 20 == 0:k
    #     plot_generated_images(e, generator)


@tf.function
def train_step(images, cap_vectors, fake_cap_vectors, generator, gen_optimizer, discriminator, disc_optimizer):
    noise = tf.random.normal([cap_vectors.shape[0], noise_dim])
    inputs = tf.concat([noise, cap_vectors], axis=-1)

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        generated_images = generator(inputs, training=True)

        real_output = discriminator([images, cap_vectors])
        fake_image_output = discriminator([generated_images, cap_vectors])
        fake_text_output = discriminator([images, fake_cap_vectors])

        gen_loss = generator_loss(fake_image_output)
        disc_loss = discriminator_loss(real_output, fake_image_output, fake_text_output)

    gen_gradients = g_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = d_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

