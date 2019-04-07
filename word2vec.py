import numpy as np
from gensim.models import word2vec

# file names
data_path = "Dataset/"
feature_path = "Features/"
model_path = "Models/"
data_name = "ASD_transcript_sens.csv"
vec_name = "ASD_transcript.vector"
feature_name = "ASD_transcript.feature"
model_name = "word2vec.model"

#################################################
# Use word2vec to get the vector of each word
#################################################

# load text from transcript data
corpus = word2vec.Text8Corpus(data_path + data_name)

# train word2vec model
model = word2vec.Word2Vec(corpus, size=200, window=5, min_count=5)

# save the model to file
model.save(model_path + model_name)

# save the vector to file
model.wv.save_word2vec_format(feature_path + vec_name, binary=False)

#################################################
# Extract features according to the vector
#################################################

# create word dict which maps each word to its vector
word_dict = {}
with open (feature_path + vec_name) as infile:
    infile.readline()
    for line in infile:
        words = line.split()
        word_name = words[0]
        word_vec = np.array([float(words[i]) for i in range(1, len(words))])
        word_dict[word_name] = word_vec

# translate each line in transcript to a vector 
features = []
with open(data_path + data_name) as infile:
    for line in infile:
        words = line.split()
        feature = np.zeros(200)
        for word in words:
            feature += word_dict[word]
        features.append(feature)

# save features to file
with open(feature_path + feature_name, 'w+') as outfile:
    for feature in features:
        for item in feature:
            outfile.write("%f " % item)
        outfile.write("\n")
