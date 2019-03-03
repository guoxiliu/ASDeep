import numpy as np
import pandas as pd


# Process feature information
with open("Dataset/ASD_expression_dataset.csv") as f:
    ncols = len(f.readline().split(','))
print(ncols)

features = np.genfromtxt("Dataset/ASD_expression_dataset.csv", delimiter=',', usecols=range(3, ncols), skip_header=1)
print(features.shape)
np.savetxt("Dataset/ASD_expression_features.csv", features, delimiter=',', fmt='%.6f')


# Process label information
tmp = np.genfromtxt("Dataset/ASD_expression_dataset.csv", dtype=None, delimiter=',', usecols=2, skip_header=1, encoding="ascii")
my_dict = {'"ASD"': 1, '"Disease"': 0}
labels = [my_dict[i] for i in tmp]
print(len(labels))
np.savetxt("Dataset/ASD_expression_labels.csv", labels, fmt="%i", delimiter=',')
