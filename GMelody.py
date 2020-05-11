
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
#       outputs the results. The logging functions are commented,
#       if needed just uncomment them
#
#   -----------------------------------------------------------------

from __future__ import print_function, division

import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adagrad

from MidiCoordinator import MidiCoordinator
from Logger import Logger

import matplotlib.pyplot as plt
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

        # For this application Adagrad works great with two distinct learning rates for the discriminator and the combined model
        optimizer_discriminator = Adagrad(0.00002)
        optimizer_combined = Adagrad()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_discriminator,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates midis
        z = Input(shape=(self.latent_dim,))
        result = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated midis as input and determines validity
        validity = self.discriminator(result)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_combined)


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

        # Starting the logging session
        l = Logger()
        l.clean_log()
        l.start_log()
        
        # Instancing the midicoordinator class
        mc = MidiCoordinator(22,102)

        # Loading the dataset
        # It has a little bit of complexity because it performes a validity check for each midi
        # Went with this option because i generated shorter midis from longer ones

        # Path of the dataset
        path = '/content/GMelody/dataset'
        # Checking how many files are in the dataset folder
        files_number = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
        # Counter for the midis that pass the validity check and array for tracking them
        final_matrix_lenght = 0
        files_passed = np.zeros(files_number)
        # Temporary matrix used for checking the validity
        data = np.zeros((files_number, self.midi_ticks, self.midi_notes, 2))

        print("Preparing dataset...")
        for i in range(0, files_number):
            try:
                # Try to fit the midi into the matrix. If succeded, store the information in the validity array and increment the variable
                matrix = np.asarray(mc.midiToMatrix("/content/GMelody/dataset/%d.mid" % (i)))
                data[i+1] = matrix
                final_matrix_lenght = final_matrix_lenght + 1
                files_passed[i] = 1
            except Exception as e:
                # If not succeded, tell us why
                print("Unexpected error %s, midi n. %d is discarded" % (e, i))
        # First step recap
        print("Number of passed midis: %d" % (final_matrix_lenght))
        print("Data to take into definitive array:")
        print(files_passed)

        # Loading the definitive data array
        data = np.zeros((final_matrix_lenght, self.midi_ticks, self.midi_notes, 2))
        c = 0
        print("Loading dataset...")
        for i in range(0, files_number):
            try:
                if files_passed[i] == 1:
                    matrix = np.asarray(mc.midiToMatrix("/content/GMelody/dataset/%d.mid" % (i)))
                    #l.log_matrix_in_input(matrix, i)
                    data[c] = matrix
                    print("Loaded midi n. %d, with shape %s" % (i,matrix.shape))
                    c = c+1
            except:
                print("Unexpected error %s, midi n. %d is discarded" % (sys.exc_info()[0], i))
        print("Dataset loaded")
        #l.log_matrix_in_input(data)

        # Arrays to store losses and accuracy
        g_losses = []
        d_losses = []
        accuracy_array = []

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

            d_losses.append(d_loss[0])
            accuracy_array.append(100*d_loss[1])

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.randint(0, 2, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
            
            g_losses.append(g_loss)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            # If at save interval => save generated samples
            if epoch % sample_interval == 0:
                self.sample_midi(epoch, mc, l, batch_size, g_losses, d_losses, accuracy_array)

            # need to start working on the music generation
    
    def sample_midi(self, epoch,  midicoordinator, l, batch_size, g_losses, d_losses, accuracy_array):
        # Plot loss and accuracy
        plt.plot(g_losses)
        plt.plot(d_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['generator', 'discriminator'], loc='upper left')
        axes = plt.gca()
        axes.set_ylim([0,2])
        plt.savefig("lossplot.png")
        
        plt.clf()

        plt.plot(accuracy_array)
        plt.title('discriminator accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['val'], loc='upper left')
        plt.savefig("accplot.png")

        # Sample a number of midis equal to the batch_size and save them into the ./generated/ folder
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