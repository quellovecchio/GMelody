
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adagrad

import matplotlib.pyplot as plt
import tensorflow as tf

from MidiCoordinator import MidiCoordinator
from Logger import Logger
import numpy as np
import sys
import midi

import os.path


class GMelody():

    def __init__(self):

        self.midi_notes = 80
        self.midi_ticks = 880
        self.midi_shape = (self.midi_ticks, self.midi_notes, 2)
        self.latent_dim = 100

        #optimizer = Adam(0.0002, 0.5)
        optimizer = Adagrad()

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
        model.add(Dense(np.prod(self.midi_shape), activation='relu'))
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
        l.clean_log()
        l.start_log()
        # Load the dataset
        # Check the interval
        mc = MidiCoordinator(22,102)

        path = '/content/GMelody/dataset'
        num_files = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
        
        final_matrix_lenght = 0
        # This is an array which is helpful to build the definitive dataset and helps me managing a defective dataset
        # You should not use this solution in your project
        coolnessArray = np.zeros(num_files)
        data = np.zeros((num_files, self.midi_ticks, self.midi_notes, 2))

        print("Preparing dataset...")
        for i in range(0, num_files):
            try:
                matrix = np.asarray(mc.midiToMatrix("/content/GMelody/dataset/%d.mid" % (i)))
                #l.log_matrix_in_input(matrix, i)
                data[i+1] = matrix
                #print("Loaded midi n. %d, with shape %s" % (i,matrix.shape))
                final_matrix_lenght = final_matrix_lenght + 1
                coolnessArray[i] = 1
            except Exception as e:
                print("Unexpected error %s, midi n. %d is discarded" % (e, i))
        print("Number of passed midis: %d" % (final_matrix_lenght))
        print("Data to take into definitive array:")
        print(coolnessArray)

        data = np.zeros((final_matrix_lenght, self.midi_ticks, self.midi_notes, 2))
        c = 0
        print("Loading dataset...")
        for i in range(0, num_files):
            try:
                if coolnessArray[i] == 1:
                    matrix = np.asarray(mc.midiToMatrix("/content/GMelody/dataset/%d.mid" % (i)))
                    #l.log_matrix_in_input(matrix, i)
                    data[c] = matrix
                    print("Loaded midi n. %d, with shape %s" % (i,matrix.shape))
                    c = c+1
            except:
                print("Unexpected error %s, midi n. %d is discarded" % (sys.exc_info()[0], i))
        print("Dataset loaded")
        #l.log_matrix_in_input(data)

        #data = np.zeros((batch_size, self.midi_ticks, self.midi_notes, 2))


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
        i = 0
        for m in range(batch_size):
          name = "{}-{}".format(epoch, i)
          midicoordinator.matrixToMidi(gen_midi[i], name)
          #pattern = midi.read_midifile("/content/GMelody/generated/%d-%d.mid" % (epoch, i))
          #l.log_midi_pattern(pattern, epoch)
          i = i+1

if __name__ == '__main__':
    g = GMelody()
    g.train(epochs=100000, batch_size=32, sample_interval=500)