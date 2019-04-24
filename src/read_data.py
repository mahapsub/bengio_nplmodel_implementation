import numpy as np
import torch
import tensorflow as tf
import glob
import os 
from collections import Counter



# dir_path = 'data/corpora/'

train_path = 'data/corpora/wiki.train.txt'
valid_path = 'data/corpora/wiki.valid.txt'
test_path = 'data/corpora/wiki.test.txt'

class Corpus():
    def __init__(self):
        self.process() 
            # self.all_words_in_corpora
            # self.word_freq_table
            # self.vocabulary_length
            # self.validation_data
            # self.test_data
        self.num_features = 60
        self.window_length = 4
        self.batch_size = 128
        self.epochs = 1

        self.curr_batch = 0
        self.num_of_possible_batches = self.get_total_batches()

        # print('creating corpus from file')
        
        self.index_mapper = dict()
        count = 0
        for word, freq in self.word_freq_table.items():
            self.index_mapper[word] = count
            count += 1
        self.C = self.createC()
        self.inv_map = {v: k for k, v in self.index_mapper.items()}
        # print('Finished creating corpora representation from file')
        # print(num_total_words)
        print("Size of: ")
        print('\t--training-set: \t\t{}'.format(len(self.all_words_in_corpora)))
        print('\t--validation-set: \t\t{}'.format(len(self.validation_data)))
        print('\t--test-set: \t\t\t{}'.format(len(self.test_data)))
    
    def get_batch(self):
        x= []
        y= []
        if self.curr_batch != self.num_of_possible_batches:
            curr_dataset = self.all_words_in_corpora[self.curr_batch*self.batch_size:(self.curr_batch+1)*self.batch_size]
            print('cur_dataset length: {}'.format(len(curr_dataset)))
            for i in range(len(curr_dataset)-self.window_length):
                x_lst = curr_dataset[i:i+self.window_length]
                # print('x:{}'.format(x_lst))
                x.append([self.get_index_from_word(word) for word in x_lst])
                y.append(self.get_index_from_word(curr_dataset[i+self.window_length]))
                # print('y:{}'.format(curr_dataset[i+self.window_length]))
        self.curr_batch += 1
        return np.array(x),np.array(y).reshape(-1,1)
            

    def get_total_batches(self):
        return int(len(self.all_words_in_corpora)/self.batch_size)
    def createC(self):
        return np.random.rand(self.vocabulary_length, self.num_features)
            
    def process(self):
        all_words = []
        word_freq_table = Counter()
        with open(train_path, 'r') as f:
            lines = f.read().strip()
            all_lines = lines.split(' ')
            for word in all_lines:
                word_freq_table[word] +=1
                all_words.append(word)
        self.all_words_in_corpora = all_words
        self.vocabulary_length = len(word_freq_table)
        self.word_freq_table = word_freq_table
        with open(valid_path, 'r') as f:
            lines = f.read().strip()
            self.validation_data =  lines.split(' ')
        with open(test_path, 'r') as f:
            lines = f.read().strip()
            self.test_data = lines.split(' ')
    def get_word_freq(self):
        return self.word_freq_table
    def get_all_words_from_corpora(self):
        return self.all_words_in_corpora
    def get_index_from_word(self, word):
        if word in self.index_mapper:
            return self.index_mapper[word]
        else:
            return self.index_mapper['<unk>']
    def get_word_from_index(self, index):
        return self.inv_map[index]
    def get_probability(self, word):
        return self.word_freq_table[word]/self.vocabulary_length
    def get_feature_vector(self, idx):
        return self.C[int(idx), :]


def tensorflow_implementation():
    corp = Corpus()
    X, Y = corp.get_batch()
    print(X.shape)
    print(Y.shape)
    
    C = tf.Variable(corp.createC())

    x = tf.placeholder(tf.float32, [None, corp.num_features])
    y_pred = tf.placeholder(tf.float32, [None, corp.vocabulary_length])
    y_actual = tf.placeholder(tf.float32, [None, Y.shape[1]])
    
    

    # Create new weights and biases.
    weights = tf.Variable(tf.zeros([corp.num_features, corp.vocabulary_length]))
    biases = tf.Variable(tf.zeros([corp.vocabulary_length]))

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer_tanh = tf.nn.tanh(tf.matmul(x, weights) + biases)
    y_pred = tf.nn.softmax(layer_tanh)
    y_actual = tf.argmax(y_pred, axis=1)


    

    print(x)
    print(y_pred)
    print(y_actual)
    print(C)

    # C = tf.variable()



def main():    
    tensorflow_implementation()









if __name__ == "__main__":
    main()