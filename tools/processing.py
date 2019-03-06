import numpy as np

def get_text(file_name):
    text = open(file_name, 'r').read()
    return text

def write_text(file_name, text):
    f = open(file_name, 'w+')
    f.write( text )

class Dictionary:

    def __init__(self, text):
        self._keys = []

    def prep_text(self, text):
        pass

    def get_size(self):
        return len(self._keys)

    def format_element(self, element):
        return element

    def element_to_index(self, element):
        return self._dict.get( element, 'err' )

    def index_to_element(self, index):
        return self._keys[index]

    def one_hot(self, text):
        encoded = []
        
        for element in self.prep_text(text):
            one_hot = [0] * self.get_size()
            one_hot[self.element_to_index(element)] = 1
            encoded.append(one_hot)
        return np.array(encoded)

    def to_text(self, one_hots):
        indices = np.argmax( one_hots, axis=1 ).tolist()
        return "".join([self.index_to_element(idx) for idx in indices])

    def indices_to_text(self, indices):
        _indices = indices.astype(int).tolist()
        return "".join([self.index_to_element(idx) for idx in _indices])

    def making_embedded_one_hot(self, text, sequence_length, step = 1):
        len_unique_chars = self.get_size()

        prep_text = self.prep_text(text)

        input_chars = []
        output_char = []
        for i in range(0, len(prep_text) - sequence_length, step):
            input_chars.append(prep_text[i:i+sequence_length])
            output_char.append(prep_text[i+sequence_length])

        train_data = np.zeros((len(input_chars), sequence_length))
        target_data = np.zeros((len(input_chars), len_unique_chars))

        for i , each in enumerate(input_chars):
            for j, char in enumerate(each):
                train_data[i, j] = self._keys.index(char)
            target_data[i, self._keys.index(output_char[i])] = 1

        return train_data, target_data
    
    def making_full_one_hot(self, text, sequence_length, step = 1):
        len_unique_chars = self.get_size()

        prep_text = self.prep_text(text)

        input_chars = []
        output_char = []
        for i in range(0, len(prep_text) - sequence_length, step):
            input_chars.append(prep_text[i:i+sequence_length])
            output_char.append(prep_text[i+sequence_length])

        train_data = np.zeros((len(input_chars), sequence_length))
        target_data = np.zeros((len(input_chars), 1))

        for i , each in enumerate(input_chars):
            for j, char in enumerate(each):
                train_data[i, j] = self._keys.index(char)
            target_data[i, 0] = self._keys.index(output_char[i])

        return train_data, target_data


    def making_one_hot(self, text, sequence_length, step = 1):
        len_unique_chars = self.get_size()

        prep_text = self.prep_text(text)

        input_chars = []
        output_char = []
        for i in range(0, len(prep_text) - sequence_length, step):
            input_chars.append(prep_text[i:i+sequence_length])
            output_char.append(prep_text[i+sequence_length])

        train_data = np.zeros((len(input_chars), sequence_length, len_unique_chars))
        target_data = np.zeros((len(input_chars), len_unique_chars))

        for i , each in enumerate(input_chars):
            for j, char in enumerate(each):
                train_data[i, j, self._keys.index(char)] = 1
            target_data[i, self._keys.index(output_char[i])] = 1

        return train_data, target_data


class Vocabulary(Dictionary):
    def __init__(self, text):
        super(Vocabulary, self).__init__(text)

        from collections import Counter

        self._count = Counter(self.prep_text(text))

        self._keys  = list(self._count.keys())
        self._keys.append("\n")
        self._keys.sort()

        self._dict  = {}

        for idx, key in enumerate(self._keys):
            self._dict[key] = idx

        self.index2word_map = {index: word for word, index in self._dict.items()}

    def format_element(self, element):
        return element + " "

    def prep_text(self, text):
        return text.replace("\n", " \n ").split()

    def get_count(self):
        return self._count

    def indices_to_text(self, indices):
        _indices = indices.astype(int).tolist()
        text = " ".join([self.index_to_element(idx) for idx in _indices])
        print( text.replace(" \n ", "\n"))
        return text.replace(" \n ", "\n")


class Alphabet(Dictionary):
    def __init__(self, text):
        super(Alphabet, self).__init__(text)
        from collections import Counter
        self._count = Counter(list(text))
        self._keys  = list(self._count.keys())

        self._keys.sort()

        self._dict  = {}
        for idx, key in enumerate(self._keys):
            self._dict[key] = idx
        
        self.index2word_map = {index: word for word, index in self._dict.items()}

    def prep_text(self, text):
        return text

    def get_count(self):
        return self._count
