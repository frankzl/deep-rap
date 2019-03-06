import tools.processing as pre
import numpy as np

glove_path = "data/embeddings/glove.840B.300d.txt"

def get_glove(path_to_glove,word2index_map):
    embedding_weights = {}
    count_all_words = 0
    
    with open(path_to_glove,'r') as f:
        for line in f:
            vals = line.split(' ')
            word = str(vals[0])
            if word in word2index_map:                
                count_all_words += 1                                 
                coefs = np.asarray(vals[1:],dtype='float32')
                coefs /= np.linalg.norm(coefs)
                embedding_weights[word] = coefs
            if count_all_words== len(word2index_map) -1:
                break
    return embedding_weights

from enum import Enum

opt = Enum("Option", "MEAN RANDOM")

def get_embedding(words, option=opt.MEAN):
    embedding = get_glove(glove_path, words)

    unknowns = list(set(words) - set(embedding.keys()))

    mean = np.mean( np.array( list( embedding.values() ) ), axis=0 )

    for word in unknowns:
        embedding[ word ] = mean

    return embedding

def get_closest_words(embedded_word, embedding_dict, limit=10):
    
    weights = np.array(list(embedding_dict.values()))
    
    cosine_dists = np.dot( weights, embedded_word)
    ff = np.argsort(cosine_dists)[::-1][:10]
    
    words = list(embedding_dict.keys())
    index2word_map = dict( (key, value) for (key, value) in enumerate(words) )
    
    words = []
    distance = []
    for f in ff:
        words.append(index2word_map[f])
        distance.append(cosine_dists[f])
    
    return list(zip(words, distance))

def get_closest_word(embedding_word, embedding_dict):
    
    words = get_closest_words(embedding_word, embedding_dict, limit=1)
    
    return words[0][0]
