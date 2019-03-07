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
    print( f"seed: \t{seed_text}" )
        
    #predict next symbols
    for i in range(length):
        seed = encoder.encode_raw( seed_chars )
        # Take only the last required symbols
        seed = seed[:,-1*trainable.time_steps:,:]
            
        # remove_fist_char = seed[:,1:,:]
        # seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, trainable.vocab_size]), axis=1)
            
        predicted = trainable.session.run([trainable.final_output], feed_dict = {trainable.X:seed})
        predicted = np.asarray(predicted[0]).astype('float64')[0]
        
        predicted_symbol = decoder.decode( predicted )
        seed_chars += predicted_symbol
    print ('result:'+ seed_chars)


def train_model(trainable, train_data, train_labels, sampler, epochs=20, batch_size=128):
    train_losses = []
    train_accs = []
    
    trainable.session = tf.Session()
    session = trainable.session
    
    with session.as_default():
        session.run(tf.global_variables_initializer())
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
                print(f"\n\nEpoch {epoch + 1}/{epochs}")
                print(f"Loss:    \t {tr_loss}")
                print(f"Accuracy:\t {tr_acc}")
            
            
            #get on of training set as seed
            # seed_text = train_data[0]
            # seed_text = train_data[0]
            seed_text = "as real as it seems the american dream\nain't nothing but another calculated schemes\nto get us locked up"
            
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