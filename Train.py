import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def train(text_vectors, images, model, epochs, batch_size=128, lr=1e-4):
    loss_fn = tf.keras.losses.binary_crossentropy
    g_optimizer = Adam(lr)
    d_optimizer = Adam(lr)

    # Rescale -1 to 1
    images = images / 127.5 - 1.
    images = np.expand_dims(images, axis=3)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        dataset = tf.data.Dataset.from_tensor_slices((images, text_vectors))
        dataset.batch(batch_size)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_image_pred, real_image_pred, fake_caption_pred = model(dataset)
            fake_image_loss, real_image_loss, fake_caption_loss = loss_fn(fake, fake_image_pred), loss_fn(valid, real_image_pred), loss_fn(fake, fake_caption_pred)
            d_loss = (fake_image_loss + real_image_loss + fake_caption_loss) / 3
            g_loss = loss_fn(valid, fake_image_pred)
        g_grads = d_tape.gradient(d_loss, model.text_encoder.trainable_variables + model.generator.trainable_variables)
        d_grads = g_tape.gradient(g_loss, model.discriminator.trainable_variables)
        g_optimizer.apply_gradients([g_grads, model.text_encoder.trainable_variables + model.generator.trainable_variables])
        d_optimizer.apply_gradients([d_grads, model.discriminator.trainable_variables])

        # Plot the progress
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

# class GAN():
#     def __init__(self):
#         self.img_rows = 64
#         self.img_cols = 64
#         self.channels = 3
#         self.img_shape = (self.img_rows, self.img_cols, self.channels)
#         self.latent_dim = 100
#
#         optimizer = Adam(0.0001, 0.9)
#
#         # Build and compile the discriminator
#         self.discriminator = self.DNet()
#         self.discriminator.compile(loss='binary_crossentropy',
#             optimizer=optimizer,
#             metrics=['accuracy'])
#
#         # Build the generator
#         self.generator = self.GNet()
#
#         # The generator takes noise as input and generates imgs
#         z = Input(shape=(self.latent_dim,))
#         img = self.generator(z)
#
#         # For the combined model we will only train the generator
#         self.discriminator.trainable = False
#
#         # The discriminator takes generated images as input and determines validity
#         validity = self.discriminator(img)
#
#         # The combined model  (stacked generator and discriminator)
#         # Trains the generator to fool the discriminator
#         self.combined = Model(z, validity)
#         self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
#
#
#
#     def train(self, epochs, batch_size=128):
#         # Load the dataset
#         (X, Y), (_, _) =  # 데이터 로드
#
#         # Rescale -1 to 1
#         X = X / 127.5 - 1.
#         X = np.expand_dims(X, axis=3)
#
#         # Adversarial ground truths
#         valid = np.ones((batch_size, 1))
#         fake = np.zeros((batch_size, 1))
#
#         for epoch in range(epochs):
#
#             # ---------------------
#             #  Train Discriminator
#             # ---------------------
#
#             # Select a random batch of images
#             idx = np.random.randint(0, X.shape[0], batch_size)
#             imgs = X[idx]
#
#             noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
#
#             # Generate a batch of new images
#             gen_imgs = self.generator.predict(noise)
#
#             # Train the discriminator
#             d_loss_real = self.discriminator.train_on_batch(imgs, valid)
#             d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
#             d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#
#             # ---------------------
#             #  Train Generator
#             # ---------------------
#
#             noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
#
#             # Train the generator (to have the discriminator label samples as valid)
#             g_loss = self.combined.train_on_batch(noise, valid)
#
#             # Plot the progress
#             print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
