import numpy as np
# import torchs
import tensorflow as tf
import glob
import os
from collections import Counter
import sys
import time

# dir_path = 'data/corpora/'

data_path = 'data/corpora/brown.txt'

def prepare_brown(data_path):
    vocab_size = 16383
    all_words = []
    word_freq_table = Counter()
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.read().strip()
        all_lines = lines.split(' ')
        for word in all_lines:
            # myset.add(word)
            word_freq_table[word] +=1
            all_words.append(word)
        print(len(all_words))
    vocabulary_counter = word_freq_table.most_common(vocab_size-1)
    print('transforming dataset')
    dataset = []

    viable_words = set()

    for key, val in vocabulary_counter:
        viable_words.add(key)
    
    print('viable_words len:{}'.format(len(viable_words)))
    for word in all_words:
        if word in viable_words:
            dataset.append(word)
        else:
            dataset.append('<unk>')
    
    train = dataset[:800000]
    valid = dataset[800000:1000000]
    test = dataset[1000000:]

    print('train size: {}'.format(len(train)))
    print('valid size: {}'.format(len(valid)))
    print('test size: {}'.format(len(test)))
    
    train_path = 'data/corpora/brown.train.txt'
    valid_path = 'data/corpora/brown.valid.txt'
    test_path = 'data/corpora/brown.test.txt'
    with open(train_path, 'w') as f:
        for item in train:
            f.write("%s " % item)
    with open(valid_path, 'w') as f:
        for item in valid:
            f.write("%s " % item)
    with open(test_path, 'w') as f:
        for item in test:
            f.write("%s " % item)





    

def main():
    prepare_brown(data_path)










if __name__ == "__main__":
    main()
