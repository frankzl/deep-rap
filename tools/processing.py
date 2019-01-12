import numpy as np



def get_text(file_name):
    text = open(file_name, 'r').read()
    return text

class Alphabet:
    def __init__(self, text):
        from collections import Counter
        self._count = Counter(list(text))
        self._keys  = list(self._count.keys())
        self._dict  = {}
        for idx, key in enumerate(self._keys):
            self._dict[key] = idx

    def get_count(self):
        return self._count

    def get_size(self):
        return len(self._keys)

    def letter_to_index(self, letter):
        return self._dict.get( letter, 'err' )

    def index_to_letter(self, index):
        return self._keys[index]

    def one_hot(self, text):
        encoded = []
        for letter in text:
            one_hot = [0] * self.get_size()
            one_hot[self.letter_to_index(letter)] = 1
            encoded.append(one_hot)
        return np.array(encoded)

    def to_text(self, one_hots):
        indices = np.argmax( one_hots, axis=1 ).tolist()
        return "".join([self.index_to_letter(idx) for idx in indices])

    def indices_to_text(self, indices):
        _indices = indices.tolist()
        return "".join([self.index_to_letter(idx) for idx in _indices])


    def making_one_hot(self, text, sequence_length, step = 1):
        '''
        '''
        len_unique_chars = self.get_size()

        input_chars = []
        output_char = []
        for i in range(0, len(text) - sequence_length, step):
            input_chars.append(text[i:i+sequence_length])
            output_char.append(text[i+sequence_length])

        train_data = np.zeros((len(input_chars), sequence_length, len_unique_chars))
        target_data = np.zeros((len(input_chars), len_unique_chars))

        for i , each in enumerate(input_chars):
            for j, char in enumerate(each):
                train_data[i, j, self._keys.index(char)] = 1
            target_data[i, self._keys.index(output_char[i])] = 1

        return train_data, target_data
