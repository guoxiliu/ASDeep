from filenames import *
import numpy as np
from gensim.models import word2vec
from math import sqrt

#################################################
# Use word2vec to get the vector of each word
#################################################

# load text from transcript data
corpus = word2vec.Text8Corpus(data_path + sentence_file)

# train word2vec model
vec_size = 256
model = word2vec.Word2Vec(corpus, size=vec_size, window=5, min_count=5)

# save the model to file
model.save(model_path + word2vec_model_file)

# save the vector to file
model.wv.save_word2vec_format(feature_path + vec_file, binary=False)

#################################################
# Extract features according to the vector
#################################################

def root_mean_square(arr):
    """
    Calculate the root mean square of a list.

    Parameters
    ---------------
    arr: list
        The list of numbers.

    Returns
    ---------------
    rms: number
        The root mean square of the given list.
    
    """
    mean_square = 0
    for x in arr:
        mean_square += x*x
    mean_square /= len(arr)
    return sqrt(mean_square)

# create word dict which maps each word to its vector
word_dict = {}
with open (feature_path + vec_file) as infile:
    infile.readline()
    for line in infile:
        words = line.split()
        word_name = words[0]
        word_vec = np.array([float(words[i]) for i in range(1, len(words))])
        word_dict[word_name] = word_vec

# translate each line in transcript to a vector 
features = []
with open(data_path + sentence_file) as infile:
    for line in infile:
        words = line.split()
        feature = np.zeros(vec_size)
        for word in words:
            feature += word_dict[word]
        rms = root_mean_square(feature)
        feature = [x / rms for x in feature]
        features.append(feature)

# save features to file
with open(feature_path + word2vec_feature_file, 'w+') as outfile:
    for feature in features:
        for item in feature:
            outfile.write("%f " % item)
        outfile.write("\n")
