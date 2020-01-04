import numpy as np
from datetime import datetime

class Logger():

    def __init__(self, namefile="log.txt"):
        super().__init__()

    def log_matrix_at_epoch(self, matrix, n=0):
        out_file = open("log.txt","a")
        out_file.write("Output matrix for epoch n. %d \n\n" % (n))
        out_file.write(np.array2string(matrix, threshold=1000, edgeitems=1000))
        out_file.write("\n\n")
        out_file.close()

    def log_matrix_in_input(self, matrix, n=0):
        out_file = open("log.txt","a")
        out_file.write("Input matrix n. %d \n\n" % (n))
        out_file.write(np.array2string(matrix, threshold=1000, edgeitems=1000))
        out_file.write("\n\n")
        out_file.close()

    def log_midi_pattern(self, pattern):
        out_file = open("log.txt","a")
        out_file.write("Associated pattern: %d \n\n")
        out_file.write(str(pattern))
        out_file.write("\n\n")
        out_file.close()
        

    def start_log(self):
        out_file = open("log.txt","a")
        out_file.write("\n\n\n")
        out_file.write("---------------------------------------------------------\n---------------------------------------------------------\n---------------------------------------------------------\n")
        dt = datetime.now().strftime("%I:%M%p on %B %d, %Y")
        out_file.write("Log of the %s execution" % (dt))
        out_file.write("\n\n\n")

    def clean_log(self):
        out_file = open("log.txt","w")
        out_file.write("")
        out_file.close()
