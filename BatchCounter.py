from MidiCoordinator import MidiCoordinator
import numpy as np
import sys


batch_size = 119
c = 0
mc = MidiCoordinator(24,102)
for i in range(0, batch_size):
    try:
        matrix = np.asarray(mc.midiToMatrix("./dataset/%d.mid" % (i)))
        c = c + 1
        print("Loaded midi n. %d, with shape %s, number %d" % (i,matrix.shape,c))
    except:
        print("Unexpected error %s, midi n. %d is discarded" % (sys.exc_info()[0], i))
print("Done")