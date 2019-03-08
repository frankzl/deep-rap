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

def train(trainable, train_data, train_labels, alphabet, epochs=20, batch_size=128, temperature=0.5, embedding=False):
    """ takes a Trainable object and trains it on the given data

    Parameters
    ----------
    trainable: Trainable
        The model to be trained

    train_data:
        The data used for training

    train_labels:
        The labels for training

    alphabet

    """
    train_losses = []
    train_accs = []
    
    trainable.session = tf.Session()
    session = trainable.session
    
    with session.as_default():
        session.run(tf.global_variables_initializer())

        for epoch in range(epochs):

            loss_sum = 0
            acc_sum = 0
            it = 0
                        
            for batch_ixs in batch_data(len(train_data), batch_size):
                _, tr_loss, tr_acc = session.run([trainable.train_step, trainable.loss, trainable.accuracy],
                        feed_dict={
                            trainable.X: train_data[batch_ixs],
                            trainable.Y: train_labels[batch_ixs],
                            })
                loss_sum += tr_loss
                acc_sum  += tr_acc
                it += 1

            train_losses.append(loss_sum/it)
            train_accs.append(acc_sum/it)

            if(epoch + 1) % 1 == 0:
                print(f"\n\nEpoch {epoch + 1}/{epochs}")
                print(f"Loss:    \t {tr_loss}")
                print(f"Accuracy:\t {tr_acc}")

            #to print the seed characters
            seed_chars = ''

            if embedding:
                seed = train_data[:1:]
                initial_seed = seed[0]
                seed_one_hot = alphabet.indices_to_text(initial_seed)
                seed_one_hot = alphabet.one_hot(seed_one_hot)
            else:
                seed = train_data[:1:]
                initial_seed = seed[0]
                seed_one_hot = np.array(initial_seed)

            # for each in seed[0]:
            for each in seed_one_hot:
                char = alphabet._keys[np.where(each == max(each))[0][0]]
                seed_chars += alphabet.format_element(char)
            print ("Seed:" + seed_chars)
    
            #predict next 500
            for i in range(500):
                if i > 0:
                    if embedding:
                        remove_fist_char = seed[:,1:]
                        seed = np.append(remove_fist_char, np.reshape(np.argmax(probabilities), [1, 1]), axis=1)
                    else:
                        remove_fist_char = seed[:,1:,:]
                        seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, alphabet.get_size()]), axis=1)
                    
                predicted = session.run([trainable.final_output], feed_dict = {trainable.X:seed})
                predicted = np.asarray(predicted[0]).astype('float64')[0]
                probabilities = sample_from_distribution(predicted, temperature)
                predicted_chars = alphabet._keys[np.argmax(probabilities)]
                seed_chars += alphabet.format_element(predicted_chars)
            print ('Result:'+ seed_chars)
    
    trainable.hist = {
        'train_losses': np.array(train_losses),
        'train_accuracy': np.array(train_accs)
    }


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def full_layer(input, output):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, output])
    b = bias_variable([output])
    return tf.matmul(input, W) + b, W, b

def sample_from_distribution(predicted, temperature=0.5):
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
        self.embed = None

    def output(self, _inputs):
        pass

class LeanableEmbedding(Embedding):
    def __init__(self, name):
        super().__init__(name)

    def build(self, vocab_size, embedding_dimension):
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension

    def output(self, _inputs):
        embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_dimension],
                    -1.0, 1.0)
                )
        embed = tf.nn.embedding_lookup(embeddings, _inputs)
        return embed, embeddings

class Trainable:
    def __init__(self, name):
        self.name = name
        self.weights = []
        self.biases = []

        self.X = None
        self.Y = None

        self.final_output = None
        self.loss = None
        self.accuracy = None
        self.train_step = None

        self.time_steps = None

class SingleLayerRNN(Trainable):
    def __init__(self, name):
        super().__init__(name)

    def build(self, hidden_layer_size, vocab_size, time_steps, l2_reg=0.0, embedding=None):
        self.time_steps = time_steps
        self.vocab_size = vocab_size

        if(embedding is None):
            self.X = tf.placeholder(tf.float32, shape=[None, time_steps, vocab_size], name="data")
            _X = tf.transpose(self.X, [1, 0, 2])
        else:
            self.X = tf.placeholder(tf.int32, shape=[None, time_steps], name="data")

            embeddings = tf.Variable(
                    tf.random_uniform([vocab_size, embedding.embedding_dimension],
                        -1.0, 1.0)
                    )
            print(embeddings.shape)
            print(self.X.shape)
            embed = tf.nn.embedding_lookup(embeddings, self.X)
            _X = tf.transpose(embed, [1, 0, 2])

            # self.X = embed

        self.Y = tf.placeholder(tf.int16, shape=[None, vocab_size], name="labels")

        # _X = tf.transpose(self.X, [1, 0, 2])

        if(embedding is None):
            _X = tf.reshape(_X, [-1, vocab_size])
        else:
            _X = tf.reshape(_X, [-1, embedding.embedding_dimension])

        _X = tf.split(_X, time_steps, 0)

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.rnn_cell   = tf.nn.rnn_cell.LSTMCell(hidden_layer_size)

            self.outputs, _ = tf.contrib.rnn.static_rnn(self.rnn_cell, _X, dtype=tf.float32)
            self.last_rnn_output = self.outputs[-1]

            self.final_output, W_out, b_out = full_layer( self.last_rnn_output, vocab_size )

            self.weights.append(W_out)
            self.biases.append(b_out)

            self.softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.final_output,
                    labels=self.Y)
            self.cross_entropy_loss = tf.reduce_mean(self.softmax)

            self.loss = self.cross_entropy_loss

            self.optimizer = tf.train.AdamOptimizer()
            self.train_step= self.optimizer.minimize(self.loss)

            self.correct_prediction = tf.equal(tf.argmax(self.Y,1), tf.argmax(self.final_output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))*100


def lstm_layer(num_layers, hidden_layer_size):
    cells = []
    for i in range(num_layers):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_layer_size)
        cells.append(lstm_cell)

    return tf.contrib.rnn.MultiRNNCell(cells=cells, state_is_tuple=True)

class MultiLayerRNN(Trainable):
    def __init__(self, name):
        super().__init__(name)

    def build(self, num_layers, hidden_layer_size, vocab_size, time_steps, l2_reg=0.0, embedding=None):
        self.time_steps = time_steps
        self.vocab_size = vocab_size

        self.X = tf.placeholder(tf.float32, shape=[None, time_steps, vocab_size], name="data")
        self.Y = tf.placeholder(tf.int16, shape=[None, vocab_size], name="labels")

        if(embedding is None):
            self.X = tf.placeholder(tf.float32, shape=[None, time_steps, vocab_size], name="data")
        else:
            self.X = tf.placeholder(tf.int32, shape=[None, time_steps], name="data")
            self.embed, _ = embedding.output(self.X)


        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            self.stacked_cells = lstm_layer(num_layers, hidden_layer_size)

            if(embedding is None):
                self.outputs, self.states = tf.nn.dynamic_rnn(
                        self.stacked_cells, self.X, dtype=tf.float32)
            else:
                self.outputs, self.states = tf.nn.dynamic_rnn(
                        self.stacked_cells, self.embed, dtype=tf.float32)

            self.last_rnn_output = self.states[num_layers - 1][1]

            self.final_output, W_out, b_out = full_layer(self.last_rnn_output, vocab_size)

            self.weights.append(W_out)
            self.biases.append(b_out)

            self.softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.final_output,
                    labels=self.Y)
            self.cross_entropy_loss = tf.reduce_mean(self.softmax)

            self.loss = self.cross_entropy_loss

            self.optimizer = tf.train.AdamOptimizer()
            self.train_step= self.optimizer.minimize(self.loss)

            self.correct_prediction = tf.equal(tf.argmax(self.Y,1), tf.argmax(self.final_output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))*100


class SimpleMultiLayerRNN(Trainable):
    def __init__(self, name):
        super().__init__(name)

    def build(self, num_layers, hidden_layer_size, vocab_size, time_steps, l2_reg=0.0):
        self.time_steps = time_steps
        self.vocab_size = vocab_size

        self.X = tf.placeholder(tf.float32, shape=[None, time_steps, vocab_size], name="data")
        self.Y = tf.placeholder(tf.int16, shape=[None, vocab_size], name="labels")

        self.X = tf.placeholder(tf.float32, shape=[None, time_steps, vocab_size], name="data")
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            self.stacked_cells = lstm_layer(num_layers, hidden_layer_size)

            self.outputs, self.states = tf.nn.dynamic_rnn(
                    self.stacked_cells, self.X, dtype=tf.float32)
            
            self.last_rnn_output = self.states[num_layers - 1][1]

            self.final_output, W_out, b_out = full_layer(self.last_rnn_output, vocab_size)

            self.weights.append(W_out)
            self.biases.append(b_out)

            self.softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.final_output,
                    labels=self.Y)
            self.cross_entropy_loss = tf.reduce_mean(self.softmax)

            self.loss = self.cross_entropy_loss

            self.optimizer = tf.train.AdamOptimizer()
            self.train_step= self.optimizer.minimize(self.loss)

            self.correct_prediction = tf.equal(tf.argmax(self.Y,1), tf.argmax(self.final_output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))*100
