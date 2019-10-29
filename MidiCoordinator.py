
#   -----------------------------------------------------------------
#
#       GMelody     0.0
#       A Generative Adversarial Network built to generate midi
#       melodies
#
#       -------------------------------------------------------
#
#       MidiCoordinator class:
#       class whose responsibility is to encode midi files into
#       readable tables.
#       Those tables are built in this way: each row represents
#       a note, and each column a timestamp. the first 'n' 
#       columns (with n = number of timestamps of the file) are
#       the "note pressed" events (when the note is executed),
#       while the others are the "note released" events, dual
#       to the firsts.
#
#           EX:
#                   Note pressed .... Note released
#                   .
#                   .
#                   .
#                   F4# 000000000000 .... 0000000000000
#                   F4  000000000000 .... 0000000000000
#                   E4  000000000000 .... 0000000000000
#                   D#4 000001000000 .... 0000010000000
#                   D4  000000100000 .... 0000001000000
#                   C#4 000000000000 .... 0000000000000
#                   C4  000100011111 .... 0001000111111
#                   .
#                   .
#                   .
#
#   -----------------------------------------------------------------

import midi
import numpy

class MidiCoordinator(object):

    def __init__(self, lowerBound, upperBound):
        self._lowerBound = lowerBound
        self._upperBound = upperBound
        self._span = upperBound-lowerBound
        
    def midiToMatrix(self, midifile, squash=True):
        schema = midi.read_midifile(midifile)
        totalTime = [track[0].tick for track in schema]
        posns = [0 for track in schema]
        matrix = []
        time = 0
        
        state = [[0,0] for x in range(self._span)]
        matrix.append(state)
        end = False
        
        while not end:
            if time % (schema.resolution/4) == (schema.resolution / 8):
                oldstate = state
                state = [[oldstate[x][0],0] for x in range(self._span)]
                matrix.append(state)
            for i in range(len(totalTime)): #For each track
                if end:
                    break
                while totalTime[i] == 0:
                    track = schema[i]
                    pos = posns[i]
    
                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < self._lowerBound) or (evt.pitch >= self._upperBound):
                            pass
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-self._lowerBound] = [0, 0]
                            else:
                                state[evt.pitch-self._lowerBound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2, 4):
                            end = True
                            break
                    try:
                        totalTime[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        totalTime[i] = None
    
                if totalTime[i] is not None:
                    totalTime[i] -= 1
    
            if all(t is None for t in totalTime):
                break
    
            time += 1
    
        S = numpy.array(matrix)
        statematrix = numpy.hstack((S[:, :, 0], S[:, :, 1]))
        statematrix = numpy.asarray(statematrix).tolist()
        return matrix

    def matrixToMidi(self, matrix, name="example"):
        matrix = numpy.array(matrix)
        if not len(matrix.shape) == 3:
            matrix = numpy.dstack((matrix[:, :self._span], matrix[:, self._span:]))
        matrix = numpy.asarray(matrix)
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)
        tickscale = 55
        
        lastcmdtime = 0
        prevstate = [[0,0] for x in range(self._span)]
        for time, state in enumerate(matrix + [prevstate[:]]):  
            offNotes = []
            onNotes = []
            for i in range(self._span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note + self._lowerBound))
                lastcmdtime = time
            for note in onNotes:
                track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note + self._lowerBound))
                lastcmdtime = time
                
            prevstate = state
        
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
                                                                
        midi.write_midifile("{}.mid".format(name), pattern)