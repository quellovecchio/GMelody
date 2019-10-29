
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

from __future__ import print_function
from MidiCoordinator import MidiCoordinator

print("a")
mc = MidiCoordinator(0,120)
print(mc.midiToMatrix("abba.mid"))