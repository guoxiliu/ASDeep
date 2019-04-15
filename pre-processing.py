from filenames import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

#################################################
# Helper functions are defined below.
#################################################

def process_expression():
    """
    The function reads the expression dataset, ignores string values, and normalize the floating
    point numbers based on log_2(number+1).
    """
    with open(data_path + express_file, "r") as infile:
        ncols = len(infile.readline().split(','))

    # Skip first 3 columns
    features = np.genfromtxt(data_path + express_file, delimiter=',', usecols=range(3, ncols), skip_header=1)

    # Normalization using logarithm
    features = np.log2(features + 1) 

    # Save the normalized data to file
    np.savetxt(data_path + norm_express_file, features, delimiter=',', fmt='%.6f')

def process_label():
    """
    The function reads the expression data and extracts the label information, i.e., ASD will be 
    treated as positive and other diseases.
    """
    tmp = np.genfromtxt(data_path + express_file, dtype=None, delimiter=',', usecols=2, skip_header=1, encoding="ascii")
    my_dict = {'"ASD"': 1, '"Disease"': 0}
    labels = [my_dict[i] for i in tmp]
    np.savetxt(data_path + label_file, labels, fmt="%i", delimiter=',')

def process_transcript(word_count): 
    """
    The function reads the transcript dataset, and splits the sequence (simply by treating word_count genes
    as a word).

    Parameters
    ---------------
    word_count: int
        The number of genes to be considered as a word.
    """
    transcript = np.genfromtxt(data_path + transcript_file, dtype=None, delimiter=',', usecols=3, skip_header=1, encoding="ascii")

    encoding = []
    for seq in transcript:
        seq = seq[1:len(seq)-1]
        total = len(seq) // word_count
        code = ""
        for i in range(total):
            code += (seq[word_count*i:word_count*(i+1)] + ' ')
        encoding.append(code)

    # write the splitted data to file
    with open(data_path + sentence_file + "_" + str(word_count) + ".csv", "w+") as file:
        for code in encoding:
            file.write(code + '\n')

#################################################
# Main entry starts here.
#################################################

if __name__ == "__main__":
    # process_expression()
    # process_label()
    for i in range (2,7): 
        process_transcript(i)