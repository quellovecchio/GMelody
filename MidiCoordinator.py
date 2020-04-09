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
#       a midi tick, and each column a note. Each unit of the 
#       matrix is a tuple of two integers in which the first 
#       one represents if there is action on that note and in 
#       that tick, the second one the velocity of the note 
#       between 0 to 50.
#       Then you can have:
#
#       [0,0]   ->  Nothing
#       [1,n]   ->  Note on event, the note is pressed at the 
#                   velocity n
#       [1,0]   ->  Note off event, the velocity of the note 
#                   goes to 0 and the sound is ended
#
#           EX:
#               note    0     1     2     3     4     ...
#           tick
#                   .
#                   .
#                   .
#           (12)    .   [0,0] [0,0] [0,0] [0,0] [0,0] ...
#           (13)    .   [0,0] [0,0] [0,0] [0,0] [0,0] ...
#           (14)    .   [0,0] [0,0] [0,0] [1,5] [0,0] ...
#           (15)    .   [0,0] [0,0] [0,0] [0,0] [0,0] ...
#           (16)    .   [1,40][0,0] [1,0] [0,0] [0,0] ...
#           (17)    .   [0,0] [0,0] [0,0] [0,0] [0,0] ...
#           (18)    .   [0,0] [0,0] [0,0] [0,0] [0,0] ...
#                   .
#                   .
#                   .
#
#       In case the midi file has more than one track, the
#       result will be handled like an array of those matrixes
#       like the one written above.
#
#       Notes on the use of midi ticks, which are the way midi
#       calculates tempo. Later you have an explanation from the
#       documentation of the midi library used in this project:
#       "You might notice that the EndOfTrackEvent has a tick 
#       value of 1. This is because MIDI represents ticks in 
#       relative time. The actual tick offset of the MidiTrackEvent
#       is the sum of its tick and all the ticks from previous 
#       events. In this example, the EndOfTrackEvent would occur 
#       at tick 101 (0 + 100 + 1)."
#
#
#
#       Known bugs: /
#
#       Not supported: Change of time signature, multi track midi
#
#   -----------------------------------------------------------------

import midi
import numpy as np
np.set_printoptions(threshold=9999999)

class MidiCoordinator(object):

    def __init__(self, lowerBound, upperBound):
            self._lowerBound = lowerBound
            self._upperBound = upperBound
            self._span = upperBound-lowerBound

    def midiToMatrix(self, midifile):
        schema = midi.read_midifile(midifile)
        # evaluating the number of tracks
        tracksNumber = len(schema)
        # if the midi has more than one track warns that the other will be discarded
        if tracksNumber > 1:
            print("The midi track has more than one track. The others will be discarded.")

        # this cylces for each track
        for i in range(tracksNumber):
            # flag to discard the track if empty
            hasEvents = False
            # gather the number of events
            eventsNumber = len(schema[i])
            # calculates the length of the track in ticks
            ticksNumber = 0
            for f in range(eventsNumber):
                ticksNumber = ticksNumber + schema[i][f].tick
            # creates the matrix of the track
            trackMatrix = np.zeros((ticksNumber, self._span, 2))
            # getting an incremental tick counter
            tickCounter = 0
            # iterate the events of the track and update the status
            for j in range(eventsNumber):
                # takes the event from the schema
                event = schema[i][j]
                tickCounter = tickCounter + event.tick
                if isinstance(event, midi.NoteEvent):
                    if (event.pitch < self._lowerBound) or (event.pitch >= self._upperBound):
                        pass
                    else:
                        if isinstance(event, midi.NoteOffEvent) or event.velocity == 0:
                            trackMatrix[tickCounter][event.pitch-self._lowerBound] = [1,0]
                        else:
                            hasEvents = True
                            trackMatrix[tickCounter][event.pitch-self._lowerBound] = [1,event.velocity]
                else:
                    pass
            if hasEvents == True:
                return trackMatrix
            else:
                raise ValueError('The track has no events')

    def matrixToMidi(self, matrix, name = "example"):
        print(matrix[0])
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)
        tickCounter = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if (matrix[i][j] == 0).any():
                    if not(matrix[i][j] == 0).all():
                        #print("Couple found: ")
                        #print(matrix[i][j])
                        #print("Response: a zero")
                        event = midi.NoteOffEvent(tick = tickCounter, pitch = j + self._lowerBound)
                        track.append(event)
                        tickCounter = 0
                    #else: 
                        #print("Couple found: ")
                        #print(matrix[i][j])
                        #print("Response: [0, 0]")   
                else:
                    #print("Couple found: ")
                    #print(matrix[i][j])
                    #print("Response: [1, 1]")
                    #print(matrix[i][j][1])
                    velocity = int(matrix[i][j][1])
                    event = midi.NoteOnEvent(tick = tickCounter, velocity = velocity, pitch = j + self._lowerBound)
                    track.append(event)
                    tickCounter = 0
            tickCounter = tickCounter+1
        endOfTrack = midi.EndOfTrackEvent(tick=1)
        track.append(endOfTrack)
        midi.write_midifile("/content/GMelody/generated/{}.mid".format(name), pattern)


