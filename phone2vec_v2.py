# -*- coding: utf-8 -*-
import os
import math
import glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import tools.processing as pre

batch_size = 256
embedding_dimension = 10
negative_samples = 32
LOG_DIR = "logs/phone2vec_v2"
EPOCHS = 5

text = pre.get_text("data/phonem-rap-lyrics/phonem_all.txt")
sentences = [text.replace( "\n", ";" )]

# Map words to indices
word2index_map = {}
index = 0

vocab = pre.Vocabulary(sentences[0])

for sent in sentences:
    for word in sent.split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
index2word_map = vocab.index2word_map
word2index_map = vocab._dict

vocabulary_size = len(index2word_map)
print("vocab_size: {} \n".format(vocabulary_size))

# Generate skip-gram pairs
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.split()
    for i in range(1, len(tokenized_sent) - 1):
        word_context_pair = [[word2index_map[tokenized_sent[i-1]],
                              word2index_map[tokenized_sent[i+1]]],
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

# start tensorflow
tf.reset_default_graph()

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

    train_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

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
        for epoch in range(EPOCHS):
            print(f"\n\nepoch: {epoch}\n")
            
            # epoch_steps = (int(len(skip_gram_pairs)/batch_size))
            epoch_steps = 1000
            for step in range(epoch_steps):
                x_batch, y_batch = get_skipgram_batch(batch_size)
                summary, _ = sess.run([merged, train_step],
                                    feed_dict={train_inputs: x_batch,
                                                train_labels: y_batch})
                # TODO we would need global_step here in order to get a nice diagram
                #      every time we start counting from zero
                train_writer.add_summary(summary, step)
                
                if step % 100 == 0:
                    loss_value = sess.run(loss,
                                            feed_dict={train_inputs: x_batch,
                                                        train_labels: y_batch})
                    print("Loss at %d/%d: %.5f" % (step, epoch_steps, loss_value))


            saver.save(sess, os.path.join(LOG_DIR, "embeddings.ckpt"), epoch)
                
    saver.save(sess, os.path.join(LOG_DIR, "final_embeddings.ckpt"))

    # Normalize embeddings before using
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)

ref_word = normalized_embeddings_matrix[word2index_map["Y"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][0:86]
for f in ff:
    print(index2word_map[f], "\t", cosine_dists[f])

# saving embedding matrix to file
with open(os.path.join(LOG_DIR, "embedding.txt"), 'w') as f:
    for i in range(vocabulary_size):
      embed = normalized_embeddings_matrix[i, :]
      word = index2word_map[i]
      f.write('%s %s\n' % (word, ' '.join(map(str, embed))))