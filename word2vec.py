from filenames import *
import numpy as np
from gensim.models import word2vec

#################################################
# Use word2vec to get the vector of each word
#################################################

# load text from transcript data
corpus = word2vec.Text8Corpus(data_path + sentence_file)

# train word2vec model
vec_size = 128
model = word2vec.Word2Vec(corpus, size=vec_size, window=5, min_count=5)

# save the model to file
model.save(model_path + word2vec_model_file)

# save the vector to file
model.wv.save_word2vec_format(feature_path + vec_file, binary=False)

#################################################
# Extract features according to the vector
#################################################

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
        features.append(feature)

# save features to file
with open(feature_path + word2vec_feature_file, 'w+') as outfile:
    for feature in features:
        for item in feature:
            outfile.write("%f " % item)
        outfile.write("\n")
