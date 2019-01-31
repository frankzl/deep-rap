import tensorflow as tf
import numpy as np
import tools.processing as pre

text = pre.get_text("data/ref_text2.txt")
sentences = text.replace("\n", ";")
vocab = pre.Vocabulary(sentences)
embedding_dimension = 3


word2index_map = {}
index = 0

# for sent in sentences:
#     for word in sent.lower().split():
#         if word not in word2index_map:
#             word2index_map[word] = index
#             index += 1
#index2word_map = {index: word for word, index in word2index_map.items()}
index2word_map = vocab.index2word_map

word2index_map = vocab._dict

vocabulary_size = len(index2word_map)



tf.reset_default_graph()

with tf.name_scope("embeddings"):
    embeddings = tf.get_variable("embedding", shape=[vocabulary_size, embedding_dimension])

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm

saver = tf.train.Saver(var_list = {"embeddings": embeddings})

import sys

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    # saver.restore(sess, "logs/word2vec_intro/final_embeddings.ckpt")
    saver.restore(sess, "logs/word2vec_intro/embeddings.ckpt-" + sys.argv[1])

    #print(vars_in_checkpoint)
    

    print("Model restored.")
    normalized_embeddings_matrix = sess.run(normalized_embeddings)
    ref_word = normalized_embeddings_matrix[word2index_map[sys.argv[2]]]

    cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
    ff = np.argsort(cosine_dists)[::-1][0:6]
    for f in ff:
        print(index2word_map[f])
        print(cosine_dists[f])
