# GMelody

GMelody is a GAN written in Python and built with Keras and Tensorflow that generates midi files

### Getting Started

Clone the project and follow the instructions to get the project up and running.

Be sure to clone the "dense2" branch which is the most up-to-date.

```
$ git clone -b dense2 https://github.com/quellovecchio/GMelody.git
```

### Prerequisites

To run the script you will need the following software:

* [Python3](https://www.python.org/downloads/)
* [Tensorflow 2](https://www.tensorflow.org/install)
* [Keras](https://keras.io/#installation)
* [python-midi](https://github.com/vishnubob/python-midi)

Installing the last one can be a little difficult. Be sure to have the "alsa-audio" package installed and to correct a little error to make the script run with python3 (just add parenthesis to the last print function) if you're running a Linux distro.

### What to do before launching

Change the dataset path directory with your own...

```
# Path of the dataset
path = '/content/GMelody/dataset'
```

...And create the directory for the generated midis.

```
mkdir ./generated
```

### How to run

Simply launch the script from the terminal.

```
python3 GMelody.py
```

### How to work with your own dataset

Load your midis into the dataset directory (be sure they have the same length in terms of midi ticks) anche change the value of the variables in the constructor method.

```
self.midi_notes = 80
self.midi_ticks = 880
```

Remember that the midi_notes value is calculated as below:

```
midi_notes = highest_note - lowest_note
```

And that the highest and the lowest notes are specified during the instancing of the MidiCoordinator class.

```
# Instancing the midicoordinator class
mc = MidiCoordinator(22,102)
```