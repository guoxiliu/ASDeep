from filenames import *

gene_list = ['A', 'C', 'G', 'T']

#################################################
# Helper functions define here
#################################################

def backtracking(k, comb, res):
    """
    Use backtracking algorithm to find all possible combinations of length k.
    
    Parameters
    ---------------
    k: int
        Length of combination.
    comb: string
        Current combination.
    res: list
        The list that contains all possible combinations.

    Returns
    ---------------
    None
    """
    if k == 0:
        res.append(comb)
        return

    for gene in gene_list:
        comb += gene
        backtracking(k-1, comb, res)
        comb = comb[:-1]


def get_k_mer_list(k):
    """
    Get the k-mer list with a given k.

    Parameters
    ---------------
    k: int
        Length of combination.

    Returns
    ---------------
    res: list
        The list that contains all possible combinations of length k.
    """
    res = []
    backtracking(k, "", res)
    return res

#################################################
# Extract k-mer features
#################################################

# get 4-mer list
k_mer_list = get_k_mer_list(4)
k_mer_dict = {}
for i in range(len(k_mer_list)):
    k_mer_dict[k_mer_list[i]] = i

# translate the gene transcript to k-mer features
k_mer_features = []
with open(data_path + sentence_file, "r") as infile:
    for line in infile:
        words = line.split()
        feature = [0] * len(k_mer_list)
        for word in words:
            feature[k_mer_dict[word]] += 1
        feature = [x / len(words) for x in feature]
        k_mer_features.append(feature)

# save k-mer features to file
with open(feature_path + k_mer_feature_file, "w+") as outfile:
    for feature in k_mer_features:
        for item in feature:
            outfile.write("%f " % item)
        outfile.write("\n")
