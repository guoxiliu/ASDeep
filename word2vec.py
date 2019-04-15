from filenames import *
import numpy as np
from gensim.models import word2vec
from math import sqrt

#################################################
# word2vec functions are defined here.
#################################################

def root_mean_square(arr):
    """
    The function calculates the root mean square of a list.

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

def extract_word2vec_feature(word_count, vec_size):
    """
    The function gets the feature based on the vector of each word 
    acquired from word2vec model. 

    Parameters
    ---------------
    word_count: int
        The number of genes to be considered as a word.
    vec_size: int
        The size of the output vector.
    """

    input_file = data_path + sentence_file + "_" + str(word_count) + ".csv"

    # load text from transcript data
    corpus = word2vec.Text8Corpus(input_file)

    # train word2vec model
    model = word2vec.Word2Vec(corpus, size=vec_size, window=5, min_count=5)

    # translate each line in transcript to a vector 
    features = []
    with open(input_file) as infile:
        for line in infile:
            words = line.split()
            feature = np.zeros(vec_size)
            for word in words:
                feature += model.wv[word]
            rms = root_mean_square(feature)
            feature = [x / rms for x in feature]
            features.append(feature)

    # save features to file
    output_file = feature_path + word2vec_feature_file + "_" + str(word_count) + ".feature"
    with open(output_file, 'w+') as outfile:
        for feature in features:
            for item in feature:
                outfile.write("%f " % item)
            outfile.write("\n")

#################################################
# Main entry starts here.
#################################################

if __name__ == "__main__":
    for i in range(2, 7):
        extract_word2vec_feature(i, 256)
    