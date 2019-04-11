from filenames import *

combined_features = []
express_features = []
word2vec_features = []

# read expression feature file
with open(feature_path + express_feature_file, "r") as infile:
    for line in infile:
        feature = [float(i) for i in line.split()]
        express_features.append(feature)


# read word2vec feature file
with open(feature_path + word2vec_feature_file, "r") as infile:
    for line in infile:
        feature = [float(i) for i in line.split()]
        word2vec_features.append(feature)

combined_features = [a + b for a,b in zip(express_features, word2vec_features)]

# save combined features to file
with open(feature_path + combined_feature_file, 'w+') as outfile:
    for feature in combined_features:
        for item in feature:
            outfile.write("%f " % item)
        outfile.write("\n")
