import numpy as np
# import torchs
import tensorflow as tf
import glob
import os
from collections import Counter
import sys
import time

# dir_path = 'data/corpora/'

train_path = '../data/corpora/brown.train.txt'


class Corpus():
    def __init__(self, name='default', features=60, window_length=4, epochs=21, h=50, batch_size=1024, path='../data/corpora/brown.train.txt'):
        self.path = path
        self.process()
        self.debug_set = set()
            # self.all_words_in_corpora
            # self.word_freq_table
            # self.vocabulary_lengthsrrrrr
            # self.validation_data
            # self.test_data
        self.num_features = features
        self.window_length = window_length-1
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_hidden_units = h
        self.name = name
        

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
        print('\t{0}: \t\t{1}'.format(self.path, len(self.all_words_in_corpora)))
        # print('\t--validation-set: \t\t{}'.format(len(self.validation_data)))
        # print('\t--test-set: \t\t\t{}'.format(len(self.test_data)))

    def get_batch(self):
        x= []
        y= []
        if self.curr_batch != self.num_of_possible_batches:
            if self.curr_batch == self.num_of_possible_batches:
                curr_dataset = self.all_words_in_corpora[self.curr_batch*self.batch_size:]
            else:
                curr_dataset = self.all_words_in_corpora[self.curr_batch*self.batch_size:(self.curr_batch+1)*self.batch_size]
            # print('cur_dataset length: {}'.format(len(curr_dataset)))
            for i in range(len(curr_dataset)-self.window_length):
                x_lst = curr_dataset[i:i+self.window_length]
                for word in x_lst:
                    self.debug_set.add(word)
                # print('x:{}'.format(x_lst))
                x.append([self.get_index_from_word(word) for word in x_lst])
                y.append(self.get_index_from_word(curr_dataset[i+self.window_length]))
                # print('y:{}'.format(curr_dataset[i+self.window_length]))
        self.curr_batch += 1
        # print(len(self.debug_set))
        if self.curr_batch == self.num_of_possible_batches:
            self.curr_batch = 0
            self.debug_set = set()

        return np.array(x),np.array(y).reshape(-1,1)


    def get_total_batches(self):
        return int(len(self.all_words_in_corpora)/self.batch_size)
    def createC(self):
        print('Creating new C')
        return np.random.rand(self.vocabulary_length, self.num_features)

    def process(self):
        all_words = []
        word_freq_table = Counter()
        with open(self.path, 'r', encoding='utf8') as f:
            lines = f.read().strip()
            all_lines = lines.split(' ')
            # myset = set()
            # for word in all_lines[:12000]:
            for word in all_lines:
                # myset.add(word)
                word_freq_table[word] +=1
                all_words.append(word)
        self.all_words_in_corpora = all_words
        self.vocabulary_length = len(word_freq_table)
        self.word_freq_table = word_freq_table
        # with open(valid_path, 'r', encoding='utf8') as f:
        #     lines = f.read().strip()
        #     self.validation_data =  lines.split(' ')
        # with open(test_path, 'r', encoding='utf8') as f:
        #     lines = f.read().strip()
        #     self.test_data = lines.split(' ')
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


def tensorflow_implementation(name_of_model):
    
    corp = None
#  def __init__(self, name='default', features=60, window_length=4, epochs=1, h=50, batch_size=1024):
    if name_of_model == 'mlp1':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp1', h=50,features=60, window_length=5, batch_size=1024)
    if name_of_model == 'mlp3':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp3', h=0,features=60, window_length=5, batch_size=1024)
    if name_of_model == 'mlp5':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp5', h=50,features=30, window_length=5, batch_size=1024)
    if name_of_model == 'mlp7':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp7', h=50,features=30, window_length=3, batch_size=1024)
    if name_of_model == 'mlp9':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp9', h=100,features=30, window_length=5, batch_size=1024)
        

    x_indexes = tf.placeholder(tf.int64, [None, corp.window_length], name='x_indexes')
    C = tf.Variable(corp.createC(), name='C')
    X = tf.reshape(tf.nn.embedding_lookup(params=C,ids=x_indexes), [-1, corp.num_features*corp.window_length], name='elookup')
    y_actual = tf.placeholder(tf.int64, [None, 1], name='y_actual')


    # Create new weights and biases.

    H = tf.Variable(tf.truncated_normal([corp.num_features*corp.window_length, corp.num_hidden_units], dtype=tf.float64), name='H')
    d = tf.Variable(tf.truncated_normal([corp.num_hidden_units], dtype=tf.float64), name='d')

    # corp.num_features*corp.window_length
    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.

    layer_tanh = tf.nn.tanh(tf.matmul(X, H) + d)

    U = tf.Variable(tf.truncated_normal([corp.num_hidden_units,corp.vocabulary_length], dtype=tf.float64), name='U')
    b = tf.Variable(tf.truncated_normal([corp.vocabulary_length], dtype=tf.float64), name='b')

    if corp.name=='mlp9':
        W = tf.Variable(tf.zeros([corp.num_features*corp.window_length, corp.vocabulary_length], dtype=tf.float64), name='W')
    else:
        W = tf.Variable(tf.truncated_normal([corp.num_features*corp.window_length, corp.vocabulary_length], dtype=tf.float64), name='W')

    full_mul = b + tf.matmul(X,W) + tf.matmul(layer_tanh,U)

    y_pred_prob = tf.nn.softmax(full_mul)
    y_pred = tf.cast(tf.argmax(full_mul, axis=1),tf.int64, name='y_pred')
    onehot_tar = tf.one_hot(tf.squeeze(y_actual), corp.vocabulary_length, 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=full_mul,
                                                           labels=onehot_tar)


    cost = tf.reduce_mean(cross_entropy, name='cost')
    tf.summary.scalar("loss", cost)
    perplex = tf.exp(cost, name='perplexity')
    tf.summary.scalar("perplexity", perplex)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    correct_prediction = tf.equal(y_pred, y_actual)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar("Accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()
    # session = tf.Session()
    epochs =corp.epochs
    logs_path = 'brown_logs/{}'.format(corp.name)
    saver = tf.train.Saver()
    log_arr = []
    model_name = corp.name

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.15
    config.gpu_options.allow_growth = True
    graph_time = tf.timestamp(name=None)
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())
        program_start = time.time()
        for i in range(epochs):
            epoch_start_time = session.run(graph_time)
            avg_cost = 0
            avg_acc = 0
            for val in range(corp.get_total_batches()):
                batch_start = time.time()
                x_batch, y_true_batch = corp.get_batch()
                feed_dict_train = {
                    x_indexes: x_batch,
                    y_actual: y_true_batch
                }
                batch_start_time = session.run(graph_time)
                _,cst, acc, summary= session.run([optimizer,cost,accuracy, merged_summary_op], feed_dict=feed_dict_train)
                cur_batch = i*corp.get_total_batches()+val
                summary_writer.add_summary(summary,cur_batch)
                avg_cost += cst / corp.get_total_batches()
                avg_acc += acc / corp.get_total_batches()
                batch_end_time = session.run(graph_time)
                if val % int(corp.get_total_batches()/10) == 0:
                    print('batch num: {0}/{1}'.format(val, corp.get_total_batches(), batch_end_time-batch_start_time))
                

            if i%3==0 or i == epochs-1:
                path_to_model= 'brown_model/{0}/model_chkpnts_{1}/'.format(model_name,str(i))
                os.makedirs(path_to_model, exist_ok=True)
                name = 'bengio'
                saver.save(session, path_to_model+name, global_step=i)

            # print(session.run(C[0,:]))
            # print(len(corp.debug_set))
            log_arr.append([avg_cost,avg_acc,i])
            np.savetxt('summary_{}.txt'.format(corp.name), np.array(log_arr))
            epoch_end_time = session.run(graph_time)
            print('epoch {0}/{3} ---- acc:{1}, cost:{2}, time: {4:.2f}'.format(i, avg_acc,avg_cost, corp.epochs, epoch_end_time-epoch_start_time))
        end = time.time()
        print('{0} model completed in ---time:{1}'.format(corp.name, end-program_start))

def select_corpus(name_of_model, corpus_path):
    corp = None
    if name_of_model == 'mlp1':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp1', h=50,features=60, window_length=5, batch_size=1024)
    if name_of_model == 'mlp3':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp3', h=0,features=60, window_length=5, batch_size=1024)
    if name_of_model == 'mlp5':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp5', h=50,features=30, window_length=5, batch_size=1024)
    if name_of_model == 'mlp7':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp7', h=50,features=30, window_length=3, batch_size=1024)
    if name_of_model == 'mlp9':
        print('Training model{}'.format(name_of_model))
        corp = Corpus(name='mlp9', h=100,features=30, window_length=5, batch_size=1024)
    return corp
# provide chkpoint directory and number for inference using provided checkpoint
def load_model(name,chk_num):
    path = 'brown_model/{0}/model_chkpnts_{1}/'.format(name, chk_num)
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(path+'bengio-'+str(chk_num)+'.meta')
        saver.restore(session,tf.train.latest_checkpoint(path))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x_indexes:0")
        y_actual = graph.get_tensor_by_name("y_actual:0")
        perplex = graph.get_tensor_by_name("perplexity:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        cost = graph.get_tensor_by_name("cost:0")
        corp = select_corpus(name, None)
        avg_cost = 0
        avg_acc = 0
        avg_perplex = 0
        for val in range(corp.get_total_batches()):
            x_batch, y_actual_batch = corp.get_batch()
            feed_dict_train = {
                x: x_batch,
                y_actual: y_actual_batch,
            }
            cst, acc, perplex = session.run([cost,accuracy, perplex], feed_dict=feed_dict_train)
            avg_cost += cst / corp.get_total_batches()
            avg_acc += acc / corp.get_total_batches()
            avg_perplex += perplex/corp.get_total_batches()
        print('cost: {0}, acc: {1}, perplex: {2}'.format(avg_cost, avg_acc, avg_perplex))
        # print(session.run([accuracy,perplex,cost], feed_dict_train))

def clean_up(name):
    import shutil
    import glob
    # os.removedirs('logs')
    chkpoints = 'model/{}'.format(name)
    r = glob.glob(chkpoints)
    print(r)
    dir_to_remove = ['logs']+r
    # print(dir_to_remove)
    for i in dir_to_remove:
        try:
            shutil.rmtree(i)
        except Exception as e:
            pass


def main():
    # load_model('mlp1', 0)
    clean_up(sys.argv[1])
    tensorflow_implementation(sys.argv[1])
    # history= np.loadtxt('history.txt')

    # import matplotlib.pyplot as plt
    # print(history)
    # print(history.shape)
    # print(history[:,1])
    # plt.plot(history[:,1])
    # plt.show()
    #
    # load_model('./model_chkpnts_90/',90)










if __name__ == "__main__":
    main()
