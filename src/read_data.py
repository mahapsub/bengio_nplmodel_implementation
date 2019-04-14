import numpy as np
import torch
import glob
import os 
from collections import Counter



dir_path = 'data/brown/'


class Corpus():
    def __init__(self):
        self.process() # makes self.all_words_in_corpora, self.word_freq_table, self.vocabulary_length
        self.num_features = 60
        self.window_length = 4
        self.batch_size = 128
        self.epochs = 1

        self.curr_batch = 0
        self.num_of_possible_batches = self.get_total_batches()

        print('creating corpus from file')
        
        self.index_mapper = dict()
        count = 0
        for word, freq in self.word_freq_table.items():
            self.index_mapper[word] = count
            count += 1
        self.C = self.createC()
        print('Finished creating corpora representation from file')
        # print(num_total_words)
    
    def get_batch(self):
        x= []
        y= []
        if self.curr_batch != self.num_of_possible_batches:
            curr_dataset = self.all_words_in_corpora[self.curr_batch*self.batch_size:(self.curr_batch+1)*self.batch_size]
            print('cur_dataset length: {}'.format(len(curr_dataset)))
            for i in range(len(curr_dataset)-self.window_length):
                x_lst = curr_dataset[i:i+self.window_length]
                x.append([self.get_index(word) for word in x_lst])
                y.append(self.get_index(curr_dataset[i+self.window_length]))
        self.curr_batch += 1
        return x,y
            

    def get_total_batches(self):
        return int(len(self.all_words_in_corpora)/self.batch_size)
    def createC(self):
        return np.random.rand(self.vocabulary_length, self.num_features)
            
    def process(self):
        corpora_list = os.listdir(dir_path)
        word_freq_table = Counter()
        word_freq_table['<unk>'] = 0
        all_words = []
        for file in corpora_list:
            if file not in ['CONTENTS', 'README', 'cats.txt', 'categories.pickle']:
                path_to_file = dir_path + file
                with open(path_to_file, 'r') as f:
                    lines = f.read()
                    all_lines = lines.split(' ')
                    all_words = all_words + all_lines
                    for word in all_lines:
                        word_freq_table[word.split('/')[0]] +=1
        self.all_words_in_corpora = all_words
        self.vocabulary_length = len(word_freq_table)
        self.word_freq_table = word_freq_table

    def get_word_freq(self):
        return self.word_freq_table
    def get_all_words_from_corpora(self):
        return self.all_words_in_corpora
    def get_index(self, word):
        if word in self.index_mapper:
            return self.index_mapper[word]
        else:
            return self.index_mapper['<unk>']
    def get_probability(self, word):
        return self.word_freq_table[word]/self.vocabulary_length
    def get_feature_vector(self, word):
        idx = self.get_index(word)
        return self.C[idx, :]


def main():    
    corp = Corpus()
    # print(corp.get_index('the'))
    print(len(corp.get_feature_vector('the')))
    X, Y = corp.get_batch()
    print(X)




if __name__ == "__main__":
    main()