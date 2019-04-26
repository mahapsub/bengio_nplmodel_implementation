import numpy as np
# import torch
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
        self.epochs = 100

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
            # print('cur_dataset length: {}'.format(len(curr_dataset)))
            for i in range(len(curr_dataset)-self.window_length):
                x_lst = curr_dataset[i:i+self.window_length]
                # print('x:{}'.format(x_lst))
                x.append([self.get_index_from_word(word) for word in x_lst])
                y.append(self.get_index_from_word(curr_dataset[i+self.window_length]))
                # print('y:{}'.format(curr_dataset[i+self.window_length]))
        self.curr_batch += 1
        if self.curr_batch == self.num_of_possible_batches:
            self.curr_batch = 0
        return np.array(x),np.array(y).reshape(-1,1)
            

    def get_total_batches(self):
        return int(len(self.all_words_in_corpora)/self.batch_size)
    def createC(self):
        return np.random.rand(self.vocabulary_length, self.num_features)
            
    def process(self):
        all_words = []
        word_freq_table = Counter()
        with open(train_path, 'r', encoding='utf8') as f:
            lines = f.read().strip()
            all_lines = lines.split(' ')
            for word in all_lines:
                word_freq_table[word] +=1
                all_words.append(word)
        self.all_words_in_corpora = all_words[:10000]
        self.vocabulary_length = len(word_freq_table)
        self.word_freq_table = word_freq_table
        with open(valid_path, 'r', encoding='utf8') as f:
            lines = f.read().strip()
            self.validation_data =  lines.split(' ')
        with open(test_path, 'r', encoding='utf8') as f:
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
    x_indexes = tf.placeholder(tf.int64, [None, corp.window_length], name='x_indexes')
    C = tf.Variable(corp.createC(), name='C')
    X = tf.reshape(tf.nn.embedding_lookup(params=C,ids=x_indexes), [-1, corp.num_features*corp.window_length])
    y_actual = tf.placeholder(tf.float32, [None, 1], name='y_actual')
    
    
    # Create new weights and biases.
    weights = tf.Variable(tf.truncated_normal([corp.num_features*corp.window_length, corp.vocabulary_length], dtype=tf.float64), name='w1')
    biases = tf.Variable(tf.truncated_normal([corp.vocabulary_length], dtype=tf.float64), name='bias')

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.

    
    layer_tanh = tf.nn.tanh(tf.matmul(X, weights) + biases)
    y_pred_prob = tf.nn.softmax(layer_tanh)
    y_pred = tf.cast(tf.argmax(y_pred_prob, axis=1),tf.float32, name='y_pred')

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_tanh,
                                                           labels=y_actual)
    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("loss", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    correct_prediction = tf.equal(y_pred, y_actual)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("Accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()
    # session = tf.Session()
    epochs =corp.epochs
    logs_path = 'logs/'
    saver = tf.train.Saver()
    log_arr = []
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())
        for i in range(epochs):
            avg_cost = 0
            avg_acc = 0
            for val in range(corp.get_total_batches()):
                x_batch, y_true_batch = corp.get_batch()
                feed_dict_train = {
                    x_indexes: x_batch,
                    y_actual: y_true_batch
                }
                if val % 100 == 0:
                    print('batch num: {}'.format(val))
                _,cst, acc, summary = session.run([optimizer,cost,accuracy, merged_summary_op], feed_dict=feed_dict_train)
                cur_batch = i*corp.get_total_batches()+val
                summary_writer.add_summary(summary,cur_batch)
                avg_cost += cst / corp.get_total_batches()
                avg_acc += acc / corp.get_total_batches()
            if i%10==0:
                saver.save(session, './model_chkpnts_{}/bengio_run_test'.format(str(i)), global_step=i)
            
            print('epoch {0} ---- acc:{1}, cost:{2}'.format(i, avg_acc,avg_cost))
            log_arr.append([avg_cost,avg_acc,i])    
            np.savetxt('history.txt', np.array(log_arr))



def main():    
    tensorflow_implementation()
    # history= np.loadtxt('history.txt')

    # import matplotlib.pyplot as plt
    # print(history)
    # print(history.shape)
    # print(history[:,1])
    # plt.plot(history[:,1])
    # plt.show()








if __name__ == "__main__":
    main()