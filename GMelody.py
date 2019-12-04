
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
from Logger import Logger
import numpy as np
import sys


class GMelody():

    def __init__(self):

        self.midi_notes = 78
        self.midi_ticks = 17
        self.midi_shape = (self.midi_ticks, self.midi_notes)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

         # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates midis
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
        model.add(Dense(np.prod(self.midi_shape), activation='sigmoid'))
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

        l = Logger()
        l.start_log()
        # Load the dataset
        mc = MidiCoordinator(24,102)
        data = np.zeros((batch_size, self.midi_ticks, self.midi_notes))

        print("-------------------Loading dataset--------------------")

        for i in range(1, batch_size+1):
            try:
                matrix = np.asarray(mc.midiToMatrix("./dataset/%d.mid" % (i)))
                matrix = np.arange(self.midi_notes * self.midi_ticks).reshape(self.midi_notes, self.midi_ticks,)
                print("Loaded midi n. %d, with shape %s" % (i,matrix.shape))
                data[i-1] = matrix
            except:
                print("Skipped midi n. %d" % (i))

        print("-------------------Dataset loaded--------------------")

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of midis
            idx = np.random.randint(0, data.shape[0], batch_size)
            phrases = data[idx]

            noise = np.random.randint(0, 2, (batch_size, self.latent_dim))

            # Generate a batch of new midis
            gen_phrases = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(phrases, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_phrases, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.randint(0, 2, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated samples
            if epoch % sample_interval == 0:
                self.sample_midi(epoch, mc, l, batch_size)

            # need to start working on the music generation
    
    def sample_midi(self, epoch,  midicoordinator, l, batch_size):
        r, c = 5, 5
        noise = np.random.randint(0, 2, (batch_size, self.latent_dim))
        gen_midi = self.generator.predict(noise)
        gen_midi2 = np.transpose(gen_midi, [0,2,1])
        l.export_matrix(gen_midi2[:1,:,:2], epoch)
        midicoordinator.matrixToMidi(gen_midi2[:1,:,:2], epoch)

if __name__ == '__main__':
    g = GMelody()
    g.train(epochs=50000, batch_size=119, sample_interval=200)