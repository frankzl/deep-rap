import tools.processing as pre
import numpy as np
from enum import Enum

opt = Enum("Option", "MEAN RANDOM")

glove_path = "data/embeddings/glove.840B.300d.txt"
phonem_path = "data/embeddings/embedding_phonems.txt"
GLOVE_SIZE = 300
PHONEM_SIZE = 3

def get_glove(path_to_glove, wordlist):
    embedding_weights = {}
    count_all_words = 0
    
    with open(path_to_glove,'r') as f:
        for line in f:
            vals = line.split(' ')
            word = str(vals[0])
            if word in wordlist:                
                count_all_words += 1                                 
                coefs = np.asarray(vals[1:],dtype='float32')
                coefs /= np.linalg.norm(coefs)
                embedding_weights[word] = coefs
            if count_all_words== len(wordlist) -1:
                break
    return embedding_weights

def get_embedding(words, option=opt.MEAN):
    embedding = get_glove(glove_path, words)

    unknowns = list(set(words) - set(embedding.keys()))

    mean = np.mean( np.array(list(embedding.values())), axis=0 )

    for word in unknowns:
        embedding[word] = mean

    return embedding

def get_embedding_matrix(word2index_map, vocabulary_size):
    wordlist = list(word2index_map.keys())
    word2embedding_dict = get_embedding(wordlist)
    embedding_matrix = np.zeros((vocabulary_size, GLOVE_SIZE))

    for word, index in word2index_map.items():
        if not word == "PAD_TOKEN":
            word_embedding = word2embedding_dict[word]
            embedding_matrix[index, :] = word_embedding

    return embedding_matrix

def get_phonem(path_to_phonem, phonemlist):
    embedding_weights = {}
    count_all_words = 0
    
    with open(path_to_phonem, 'r') as f:
        for line in f:
            vals = line.split(' ')
            word = str(vals[0])
            if word in phonemlist:                
                count_all_words += 1                                 
                coefs = np.asarray(vals[1:],dtype='float32') # already normalized
                embedding_weights[word] = coefs
            if count_all_words == len(phonemlist) -1:
                break
    return embedding_weights

def get_phonem_embedding(phonems):
    embedding = get_phonem(phonem_path, phonems)

    unknowns = list(set(phonems) - set(embedding.keys()))

    mean = np.mean( np.array(list(embedding.values())), axis=0 )

    for phonem in unknowns:
        embedding[phonem] = mean

    return embedding

def get_phonem_embedding_matrix(phonem2index_map, vocabulary_size):
    phonemlist = list(phonem2index_map.keys())
    word2embedding_dict = get_phonem_embedding(phonemlist)
    embedding_matrix = np.zeros((vocabulary_size, PHONEM_SIZE))

    for phonem, index in phonem2index_map.items():
        if not phonem == "PAD_TOKEN":
            phonem_embedding = word2embedding_dict[phonem]
            embedding_matrix[index, :] = phonem_embedding

    return embedding_matrix

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
