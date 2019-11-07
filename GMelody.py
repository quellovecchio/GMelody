
#   -----------------------------------------------------------------
#
#       GMelody     0.0
#       A Generative Adversarial Network built to generate midi
#       melodies
#
#       --------------------------------------------------------
#
#       GMelody:
#       This is the main script, it coordinates the modules and
#       outputs the results
#
#   -----------------------------------------------------------------

from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import tensorflow as tf

from MidiCoordinator import MidiCoordinator

import sys

import numpy as np

class GMelody():

    def __init__(self):

        self.midi_notes = 88
        self.midi_ticks = 17
        self.midi_shape = (self.midi_notes, self.midi_ticks)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

         # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        result = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(result)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.midi_shape), activation='tanh'))
        model.add(Reshape(self.midi_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        result = model(noise)

        return Model(noise, result)


    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.midi_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        result = Input(shape=self.midi_shape)
        validity = model(result)

        return Model(result, validity)


    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        # This is code for testing actually
        mc = MidiCoordinator(24,102)
        matrix = mc.midiToMatrix("2.mid")
        data = tf.data.Dataset.from_tensor_slices(matrix)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of midis
            #idx = np.random.randint(0, data.shape[0], batch_size)
            # 0 in the next line has to be substituted with idx, this is for testing purposes only
            phrases = np.array([data])

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_phrases = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(phrases, valid, sample_weight=None, class_weight=None, reset_metrics=True)
            d_loss_fake = self.discriminator.train_on_batch(gen_phrases, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_midi(epoch)

            # need to start working on the music generation
    
    def sample_midi(self, epoch):

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_midi = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_midi = 0.5 * gen_midi + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_midi[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("generated/%d.mid" % epoch)
        plt.close()

if __name__ == '__main__':
    g = GMelody()
    g.train(epochs=30000, batch_size=1, sample_interval=200)