import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

###################################
#   Process feature information  
###################################
with open("Dataset/ASD_expression_dataset.csv") as f:
    ncols = len(f.readline().split(','))

# Skip first 3 columns
features = np.genfromtxt("Dataset/ASD_expression_dataset.csv", delimiter=',', usecols=range(3, ncols), skip_header=1)

# Normalization using z-score
column_means = np.mean(features, axis=0)
column_stds = np.std(features, axis=0)

for i in range(features.shape[1]):
    features[:, i] = (features[:, i] - column_means[i]) / column_stds[i]

np.savetxt("Dataset/ASD_expression_features.csv", features, delimiter=',', fmt='%.6f')

###################################
# Process label information
###################################
tmp = np.genfromtxt("Dataset/ASD_expression_dataset.csv", dtype=None, delimiter=',', usecols=2, skip_header=1, encoding="ascii")
my_dict = {'"ASD"': 1, '"Disease"': 0}
labels = [my_dict[i] for i in tmp]
np.savetxt("Dataset/ASD_expression_labels.csv", labels, fmt="%i", delimiter=',')

###################################
# Process transcript information
###################################
transcript = np.genfromtxt("Dataset/ASD_transcript_seqs.csv", dtype=None, delimiter=',', usecols=3, skip_header=1, encoding="ascii")

encoding = []
for seq in transcript:
    seq = seq[1:len(seq)-1]
    total = len(seq) // 4
    code = ""
    for i in range(total):
        code += (seq[4*i:4*(i+1)] + ' ')
    encoding.append(code)

# write the splitted data to file
with open("Dataset/ASD_transcript_sens.csv", 'w') as file:
    for code in encoding:
        file.write(code + '\n')
