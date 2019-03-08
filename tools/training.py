import numpy as np
import tensorflow as tf
import glob
import os


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


def sample( seed_text, trainable, encoder, decoder, length=40 ):
    
    """ prints the sampled string
    
    seed_text: string of the seed, must have minimum length of our timestep size
    
    trainable: object model to sample from
    
    encoder: encoder object to encode the seed_text
    
    decoder: decoder object to decode the output from the trainable
    
    length: how many symbols we want to sample
    
    """
    
    seed = encoder.encode_raw( seed_text )

    #to print the seed characters
    seed_chars = seed_text
    print( "------Sampling----------" )
    print( f"seed: \n{seed_text}\n-" )
        
    #predict next symbols
    for i in range(length):
        seed = encoder.encode_raw( seed_chars )
        # Take only the last required symbols
        if len(seed.shape) == 3:
            seed = seed[:,-1*trainable.time_steps:,:]
        elif len(seed.shape) == 2:
            seed = seed[:,-1*trainable.time_steps:]

            
        # remove_fist_char = seed[:,1:,:]
        # seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, trainable.vocab_size]), axis=1)
            
        predicted = trainable.session.run([trainable.final_output], feed_dict = {trainable.X:seed})
        predicted = np.asarray(predicted[0]).astype('float64')[0]
        
        predicted_symbol = decoder.decode( predicted )
        seed_chars += predicted_symbol
    print ('result: \n'+ seed_chars)


def train_model(trainable, train_data, train_labels, sampler, epochs=20, batch_size=128, log_dir=None, 
                embedding_matrix=None):
    train_losses = []
    train_accs = []
    
    trainable.session = tf.Session()
    session = trainable.session
    
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=0.5)

    with session.as_default():
        session.run(tf.global_variables_initializer())
        # assign pretrained embedding matrix
        if embedding_matrix:
            session.run(trainable.embedding_init, feed_dict={trainable.embedding_placeholder: embedding_matrix})

        if log_dir:
            LOG_DIR = "../" + log_dir
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR, exist_ok=True)
            if glob.glob(LOG_DIR + "/*.meta"):
                saver = tf.train.import_meta_graph(glob.glob(LOG_DIR + '/*.meta')[0])
                saver.restore(session, os.path.join(LOG_DIR, "model"))
                print("Restoring an old model from '{}' and training it further..".format(LOG_DIR))
            else:
                print("Building model from scratch! \n Saving into: '{}'".format(LOG_DIR))
        else:
            print("Building model from scratch! \n Saving into: '{}'".format(LOG_DIR))

        tr_loss, tr_acc = session.run([trainable.loss, trainable.accuracy],
                                      feed_dict={trainable.X: train_data,
                                                 trainable.Y: train_labels})
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        
        for epoch in range(epochs):
             
            for batch_ixs in batch_data(len(train_data), batch_size):
                _ = session.run(trainable.train_step,
                               feed_dict={
                                   trainable.X: train_data[batch_ixs],
                                   trainable.Y: train_labels[batch_ixs],
                               })
            tr_loss, tr_acc = session.run([trainable.loss, trainable.accuracy],
                                           feed_dict={trainable.X: train_data,
                                                      trainable.Y: train_labels
                                                     })
            train_losses.append(tr_loss)
            train_accs.append(tr_acc)
            
            if(epoch + 1) % 1 == 0:
                # saving the session into "model"
                saver.save(session, os.path.join(LOG_DIR, "model"))
                print(f"\n\nEpoch {epoch + 1}/{epochs}")
                print(f"Loss:    \t {tr_loss}")
                print(f"Accuracy:\t {tr_acc}")
            
            
            #get on of training set as seed
            # seed_text = train_data[0]
            # seed_text = train_data[0]
            seed_text = """as real as it seems the american dream
ain't nothing but another calculated schemes\nto get us locked up shot up back in chains
to deny us of the future rob our names\nkept my history of mystery but now i see
the american dream wasn't meant for me\ncause lady liberty is a hypocrite she lied to me\npromised me freedom education equality
never gave me nothing but slavery\nand now look at how dangerous you made me"""
            
            sampler(trainable, seed_text)
            
    
    trainable.hist = {
        'train_losses': np.array(train_losses),
        'train_accuracy': np.array(train_accs)
    }


def sample_from_distribution(predicted, temperature=0.5):
    '''
     helper function to sample an index from a probability array
    '''
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

class Encoder:
    def __init__(self, name):
        self.name = name
    def encode(self, seed_chars):
        pass

    def encode_raw(self, text):
        pass
    
class Decoder:
    def __init__(self, name):
        self.name = name
    def decode(self, predicted):
        pass

class OneHotEncoder(Encoder):
    """
    Encodes sequences of words to sequences of 1-Hot Encoded vectors
    """
    
    def __init__(self, name, word2index):
        super(OneHotEncoder, self).__init__(name)
        self.word2index = word2index
        
    def encode(self, sequences):
        encoded_sequences = []
        for seq in sequences:
            encoded = np.zeros( ( len(seq), len(self.word2index) ) )
            
            for idx, symbol in enumerate(seq):
                encoded[idx][ self.word2index[symbol] ] = 1
            
            encoded_sequences.append(encoded)
        
        return np.array(encoded_sequences)

    def encode_raw(self, text):
        return self.encode( [text] )
    
    def encode_labels(self, labels):
        
        encoded = []
        
        for label in labels:
            one_hot_vec = np.zeros(len(self.word2index), dtype=int)
            one_hot_vec[ self.word2index[label] ] = 1
            encoded.append( one_hot_vec )
            
        return np.array(encoded)
    
class OneHotDecoder(Decoder):
    """
    Decodes a 1-Hot Encoded vector (prediction) to a word
    """
    def __init__(self, name, index2word, temperature=0.5):
        super(OneHotDecoder, self).__init__(name)
        self.temperature = temperature
        self.index2word = index2word 
        
    def decode(self, predicted):
        predicted = sample_from_distribution(predicted, temperature=self.temperature)
        return self.index2word[ np.argmax(predicted) ]

class OneHotWordEncoder(Encoder):
    """
    Encodes sequences of words to sequences of 1-Hot Encoded vectors
    """
    
    def __init__(self, name, word2index):
        super(OneHotWordEncoder, self).__init__(name)
        self.word2index = word2index
        
    def encode(self, sequences):
        """
        Encodes our sequences of words to sequences of 1-Hots
        """
        try:
            encoded_sequences = []
            for seq in sequences:
                
                encoded = np.zeros( ( len(seq), len(self.word2index) ) )
                
                for idx, word in enumerate(seq):
                    encoded[idx][ self.word2index[word] ] = 1
                
                encoded_sequences.append(encoded)
            
            return np.array(encoded_sequences)
        except Exception as e:
            print(e)
    
    def encode_raw(self, text):
        """
        Encodes a text to sequences of 1-Hots (needed for sampling)
        """
        text = text.replace("\n", " \\n ")
        text = text.replace(" +", " ")
        words = text.split(" ")
        encoded = np.zeros( ( len(words), len(self.word2index) ) )
        
        for idx, word in enumerate(words):
            if word != "":
                encoded[idx][ self.word2index[word] ] = 1
        
        return np.array( [encoded] )
        
    
    def encode_labels(self, labels):
        """
        Encodes the labels (sequences of one word)
        """
        
        encoded = []
        
        for label in labels:
            one_hot_vec = np.zeros(len(self.word2index), dtype=int)
            one_hot_vec[ self.word2index[label] ] = 1
            encoded.append( one_hot_vec )
            
        return np.array(encoded)
    
class OneHotWordDecoder(Decoder):
    """
    Decodes a 1-Hot Encoded vector (prediction) to a word
    """
    def __init__(self, name, index2word, temperature=0.5):
        super(OneHotWordDecoder, self).__init__(name)
        self.temperature = temperature
        self.index2word = index2word 
        
    def decode(self, predicted):
        predicted = sample_from_distribution(predicted, temperature=self.temperature)
        return " " + self.index2word[ np.argmax(predicted) ].replace("\\n","\n")

class IndexWordEncoder(Encoder):
    """
    Encodes sequences of words to sequences of 1-Hot Encoded vectors
    """
    
    def __init__(self, name, word2index):
        super(IndexWordEncoder, self).__init__(name)
        self.word2index = word2index
        
    def encode(self, sequences):
        """
        Encodes our sequences of words to sequences of indices
        """
        encoded_sequences = []
        for seq in sequences:
            
            # encoded = np.zeros( len(seq) )
            encoded = [ self.word2index[word] for word in seq ]
            
            encoded_sequences.append(encoded)
        
        return np.array(encoded_sequences)
    
    def encode_raw(self, text):
        """
        Encodes a text to sequences of indices (needed for sampling)
        """
        text = text.replace("\n", " \\n ")
        text = text.replace(" +", " ")
        words = text.split(" ")
        encoded = np.zeros( len(words) )
        
        for idx, word in enumerate(words):
            if word != "":
                encoded[idx] = self.word2index[word]
        
        return np.array( [encoded] )
        
    
    def encode_labels(self, labels):
        """
        Encodes the labels (sequences of one word)
        """
        
        encoded = []
        
        for label in labels:
            one_hot_vec = np.zeros(len(self.word2index), dtype=int)
            one_hot_vec[ self.word2index[label] ] = 1
            encoded.append( one_hot_vec )
            
        return np.array(encoded)
