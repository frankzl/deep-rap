import numpy as np
import tensorflow as tf

def batch_data(num_data, batch_size):
    """ Yield batches with indices until epoch is over.

    Parameters
    ----------
    num_data: int
        The number of samples in the dataset.
    batch_size: int
        The batch size used using training.

    Returns
    -------
    batch_ixs: np.array of ints with shape [batch_size,]
        Yields arrays of indices of size of the batch size until the epoch is over.
    """

    # data_ixs = np.random.permutation(np.arange(num_data))
    data_ixs = np.arange(num_data)
    ix = 0
    while ix + batch_size < num_data:
        batch_ixs = data_ixs[ix:ix+batch_size]
        ix += batch_size
        yield batch_ixs


def sample(predicted, temperature=0.5):
    '''
     helper function to sample an index from a probability array
    '''
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

class Embedding:
    def __init__(self, name):
        self.name = name

    def build(self, vocab_size, embedding_dimension):

        self._inputs = tf.placeholder(tf.int32, shape=[-1, times_steps])

        self.embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_dimension],
                    -1.0, 1.0)
                )
        self.embed = tf.nn.embedding_lookup(self.embeddings, self._inputs)

        
class SingleLayerRNN:
    def __init__(self, name):
        self.name = name
        self.weights = []
        self.biases = []
        
    def build(self, hidden_layer_size, vocab_size, time_steps, l2_reg=0.0):
        self.time_steps = time_steps
        self.vocab_size = vocab_size
        
        self.X = tf.placeholder(tf.float32, shape=[None, time_steps, vocab_size], name="data")
        self.Y = tf.placeholder(tf.int16, shape=[None, vocab_size], name="labels")
        
        _X = tf.transpose(self.X, [1, 0, 2])
        _X = tf.reshape(_X, [-1, vocab_size])
        _X = tf.split(_X, time_steps, 0)
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.rnn_cell   = tf.nn.rnn_cell.LSTMCell(hidden_layer_size)
            
            self.outputs, _ = tf.contrib.rnn.static_rnn(self.rnn_cell, _X, dtype=tf.float32)
            
            W_out = tf.Variable(tf.truncated_normal([hidden_layer_size, vocab_size], 
                                                 mean=0, stddev=.01))
            b_out = tf.Variable(tf.truncated_normal([vocab_size],
                                                mean=0, stddev=.01))
            
            self.weights.append(W_out)
            self.biases.append(b_out)
            
            self.last_rnn_output = self.outputs[-1]
            self.final_output    = self.last_rnn_output @ W_out + b_out
            
            self.softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.final_output,
                                                                labels=self.Y)
            self.cross_entropy_loss = tf.reduce_mean(self.softmax)
            
            self.loss = self.cross_entropy_loss
            
            self.optimizer = tf.train.AdamOptimizer()
            self.train_step= self.optimizer.minimize(self.loss)
            
            self.correct_prediction = tf.equal(tf.argmax(self.Y,1), tf.argmax(self.final_output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))*100
    
    def train(self, train_data, train_labels, alphabet, epochs=20, batch_size=128, temperature=0.5):
        train_losses = []
        train_accs = []
        
        self.session = tf.Session()
        session = self.session
        
        with session.as_default():
            session.run(tf.global_variables_initializer())
            tr_loss, tr_acc = session.run([self.loss, self.accuracy],
                                          feed_dict={self.X: train_data,
                                                     self.Y: train_labels})
            train_losses.append(tr_loss)
            train_accs.append(tr_acc)
            
            for epoch in range(epochs):
                
                if(epoch + 1) % 1 == 0:
                    print(f"\n\nEpoch {epoch + 1}/{epochs}")
                    print(f"Loss:    \t {tr_loss}")
                    print(f"Accuracy:\t {tr_acc}")
                
                for batch_ixs in batch_data(len(train_data), batch_size):
                    _ = session.run(self.train_step,
                                   feed_dict={
                                       self.X: train_data[batch_ixs],
                                       self.Y: train_labels[batch_ixs],
                                   })
                tr_loss, tr_acc = session.run([self.loss, self.accuracy],
                                               feed_dict={self.X: train_data,
                                                          self.Y: train_labels
                                                         })
                train_losses.append(tr_loss)
                train_accs.append(tr_acc)
                
                #get on of training set as seed
                seed = train_data[:1:]
        
                #to print the seed 40 characters
                seed_chars = ''
                for each in seed[0]:
                    char = alphabet._keys[np.where(each == max(each))[0][0]]
                    seed_chars += alphabet.format_element(char)
                print ("Seed:" + seed_chars)
        
                #predict next 500 characters
                for i in range(500):
                    if i > 0:
                        remove_fist_char = seed[:,1:,:]
                        seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, self.vocab_size]), axis=1)
                        
                    predicted = session.run([self.final_output], feed_dict = {self.X:seed})
                    predicted = np.asarray(predicted[0]).astype('float64')[0]
                    probabilities = sample(predicted, temperature)
                    predicted_chars = alphabet._keys[np.argmax(probabilities)]
                    seed_chars += alphabet.format_element(predicted_chars)
                print ('Result:'+ seed_chars)
        
        self.hist = {
            'train_losses': np.array(train_losses),
            'train_accuracy': np.array(train_accs)
        }
