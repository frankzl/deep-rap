# -*- coding: utf-8 -*-
import os
import math
import glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import tools.processing as pre

batch_size = 256
# embedding_dimension = 128
embedding_dimension = 10
negative_samples = 32
# window_size of 1 had the best outcome to predict words which have an equal meaning, 
# window_size of 5 predicted topical words which we don't need in our case
# For the best results we decided to use the GloVe embeddings
window_size = 1 
so we have decided to take the GloVe embeddings
LOG_DIR = "logs/word2vec_v2"
EPOCHS = 10

tf.reset_default_graph()

text = pre.get_text("data/cleaned-rap-lyrics/final_2_pac_rakim_kid_cudi.txt")

sentences = []

# Create two kinds of sentences - sequences of odd and even digits.
# for i in range(10000):
#     rand_odd_ints = np.random.choice(range(1, 10, 2), 3)
#     sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
#     rand_even_ints = np.random.choice(range(2, 10, 2), 3)
#     sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))

# text = pre.get_text("data/cleaned-rap-lyrics/lyrics_combined.txt")
# text = pre.get_text("data/prepped/clean2_pac.txt")

# sentences = text.split("\n")
sentences = [text.replace( "\n", ";" )]

# Map words to indices
word2index_map = {}
index = 0

print(sentences[0][:300])

vocab = pre.Vocabulary(sentences[0])

index2word_map = vocab.index2word
word2index_map = vocab._dict

vocabulary_size = len(index2word_map)

# Generate skip-gram pairs
skip_gram_pairs = []
for sent in sentences:
    for i in range(window_size):
        sent = "EOS " + sent
        sent = sent + " EOS"

    tokenized_sent = sent.lower().split()
    for i in range(window_size, len(tokenized_sent) - window_size):
        for n in range(1, window_size + 1):
            word_context_pair = [[word2index_map[tokenized_sent[i-n]],
                                 word2index_map[tokenized_sent[i+n]]],
                                 word2index_map[tokenized_sent[i]]]
            skip_gram_pairs.append([word_context_pair[1],
                                    word_context_pair[0][0]])
            skip_gram_pairs.append([word_context_pair[1],
                                    word_context_pair[0][1]])

def get_skipgram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y

# Input data, labels
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# Embedding lookup table currently only implemented in CPU
with tf.name_scope("embeddings"):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_dimension],
                          -1.0, 1.0), name='embedding')
    # This is essentialy a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Create variables for the NCE loss
nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_dimension],
                            stddev=1.0 / math.sqrt(embedding_dimension)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels,
                 num_sampled=negative_samples, num_classes=vocabulary_size))
tf.summary.scalar("NCE_loss", loss)

# Learning rate decay
global_step = tf.Variable(0, trainable=False)
learningRate = tf.train.exponential_decay(learning_rate=0.1,
                                          global_step=global_step,
                                          decay_steps=1000,
                                          decay_rate=0.95,
                                          staircase=True)
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
merged = tf.summary.merge_all()

saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

TRAIN = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(LOG_DIR,
                                         graph=tf.get_default_graph())

    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))

    if glob.glob(LOG_DIR + '/*.meta'):
        TRAIN = False
        saver = tf.train.import_meta_graph(glob.glob(LOG_DIR + '/*.meta')[0])
        saver.restore(sess, os.path.join(LOG_DIR, "final_embeddings.ckpt"))
        # global_step = sess.run(global_step)
        print("Restoring an old model and training it further..")
    else:
        print("Building model from scratch!")
        # global_step = 0

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(train_writer, config)

    if TRAIN:
        global_step = 0
        for epoch in range(EPOCHS):

            print(f"\n\nepoch: {epoch}\n")
            
            # epoch_steps = (int(len(skip_gram_pairs)/batch_size))
            epoch_steps = 1000
            for step in range(epoch_steps):
                x_batch, y_batch = get_skipgram_batch(batch_size)
                summary, _ = sess.run([merged, train_step],
                                    feed_dict={train_inputs: x_batch,
                                                train_labels: y_batch})
                train_writer.add_summary(summary, step + global_step)
                
                if step % 100 == 0:
                    loss_value = sess.run(loss,
                                            feed_dict={train_inputs: x_batch,
                                                        train_labels: y_batch})
                    print("Loss at %d/%d: %.5f" % (step, epoch_steps, loss_value))
            
            global_step += epoch_steps
            saver.save(sess, os.path.join(LOG_DIR, "embeddings.ckpt"), epoch)
                
    saver.save(sess, os.path.join(LOG_DIR, "final_embeddings.ckpt"))

    # Normalize embeddings before using
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)

ref_word = normalized_embeddings_matrix[word2index_map["walk"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][0:50]
for f in ff:
    print(index2word_map[f])
    print(cosine_dists[f])
