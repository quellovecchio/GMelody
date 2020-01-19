from MidiCoordinator import MidiCoordinator

mc = MidiCoordinator(24,102)
matrix = mc.midiToMatrix("b.mid")
mc.matrixToMidi(matrix, "generated")