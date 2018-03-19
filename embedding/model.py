import tensorflow as tf
import time
import numpy as np
import math


class Word2Vec:
    def __init__(self, sentences, size=100, learning_rate=0.001, window=5, min_count=5, batch_size=64,
                 sample=1e-3, seed=1, epoch_num=5):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.batch_size = batch_size
        self.sample = sample
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate

        tf.set_random_seed(seed)

        self.build_vocabulary(sentences)
        self.build_network()
        self.init_op()
        self.train(sentences)


    def build_vocabulary(self, sentences):
        print 'build vocabulary...'
        self.vocabulary = {}
        for sentence in sentences:
            for word in sentence:
                if not self.vocabulary.has_key(word):
                    self.vocabulary[word] = 1
                else:
                    self.vocabulary[word] += 1
        unk_words = []
        unk_value = 0
        for key, value in self.vocabulary.items():
            if value < self.min_count:
                unk_words.append(key)
                unk_value += value
        for word in unk_words:
            self.vocabulary.pop(word)
            self.vocabulary['<unk>'] = unk_value
        for i, key in enumerate(self.vocabulary.keys()):
            self.vocabulary[key] = i
        self.vocabulary_size = len(self.vocabulary)
        self.num_sample = int(self.vocabulary_size * self.sample)


    def train(self, sentences):
        print 'training...'
        for num in range(self.epoch_num):
            print '--- epoch %s start ---' % (num)
            losses = 0
            for count, sentence in enumerate(sentences):
                inputs = []
                labels = []
                for i in range(len(sentence)):
                    start = max(0, i - self.window)
                    end = min(len(sentence), i + self.window + 1)
                    for j in range(start, end):
                        if i == j:
                            continue
                        else:
                            id = self.vocabulary.get(sentence[i], self.vocabulary['<unk>'])
                            label = self.vocabulary.get(sentence[j], self.vocabulary['<unk>'])
                            inputs.append(id)
                            labels.append(label)

                inputs = np.array(inputs, dtype=np.int32)
                labels = np.array(labels, dtype=np.int32)
                labels = np.reshape(labels, [len(labels), 1])

                feed_dict = {self.inputs: inputs,
                             self.labels: labels}

                _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                losses += loss
                if (count + 1) % 10000 == 0:
                    print '--- epoch: %s --- training %.3f --- loss: %.3f ---' % (num, float(count + 1) / len(sentences), losses / 2000)
                    losses = 0

    def save(self, path):
        word_embeddings = self.sess.run(self.word_embeddings)
        temp = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))
        f = open(path, 'w')
        f.write(str(len(self.vocabulary)) + ' ' + str(self.size) + '\n')
        for i, embedding in enumerate(word_embeddings):
            f.write(temp[i] + ' ' + ' '.join(map(str, embedding)) + '\n')
        f.close()

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)



    def build_network(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.int32, [None])
            self.labels = tf.placeholder(tf.int32, [None, 1])
            self.word_embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.size], -1.0, 1.0))
            self.weight = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.size], stddev=1.0 / math.sqrt(self.size)))
            self.bias = tf.Variable(tf.zeros([self.vocabulary_size]))

            embed = tf.nn.embedding_lookup(self.word_embeddings, self.inputs)

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(self.weight, self.bias, self.labels, embed, self.num_sample, self.vocabulary_size))

            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            self.init = tf.global_variables_initializer()


class CrossLingualWord2Vec:
    def __init__(self, sentences1, sentences2, size=50, learning_rate=0.001, window=5, min_count=5, batch_size=128,
                 sample=1e-3, seed=1, epoch_num=1):

        assert len(sentences1) == len(sentences2)

        self.name = 'cross-lingual-word2vec'
        self.time_stamp = str(int(time.time()))
        self.size = size
        self.window = window
        self.min_count = min_count
        self.batch_size = batch_size
        self.sample = sample
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate

        tf.set_random_seed(seed)

        self.build_vocabulary(sentences1, sentences2)
        self.build_network()
        self.train(sentences1, sentences2)


    def build_vocabulary(self, sentences1, sentences2):
        print 'build vocabulary...'
        self.vocabulary = {}
        for sentence in sentences1:
            for word in sentence:
                if not self.vocabulary.has_key(word):
                    self.vocabulary[word] = 1
                else:
                    self.vocabulary[word] += 1
        for sentence in sentences2:
            for word in sentence:
                if not self.vocabulary.has_key(word):
                    self.vocabulary[word] = 1
                else:
                    self.vocabulary[word] += 1
        unk_words = []
        unk_value = 0
        for key, value in self.vocabulary.items():
            if value < self.min_count:
                unk_words.append(key)
                unk_value += value
        for word in unk_words:
            self.vocabulary.pop(word)
            self.vocabulary['<unk>'] = unk_value
        for i, key in enumerate(self.vocabulary.keys()):
            self.vocabulary[key] = i
        self.vocabulary_size = len(self.vocabulary)
        self.num_sample = int(self.vocabulary_size * self.sample)


    def build_input(self, sentence):
        inputs = []
        labels = []
        for i in range(len(sentence)):
            start = max(0, i - self.window)
            end = min(len(sentence), i + self.window + 1)
            for j in range(start, end):
                if i == j:
                    continue
                else:
                    id = self.vocabulary.get(sentence[i], self.vocabulary['<unk>'])
                    label = self.vocabulary.get(sentence[j], self.vocabulary['<unk>'])
                    inputs.append(id)
                    labels.append(label)
        return inputs, labels

    def train(self, sentences1, sentences2):
        print 'training...'
        start_time = time.time()
        for num in range(self.epoch_num):
            print '--- epoch %s start ---' % (num)
            losses = 0
            for count, sentence in enumerate(sentences1):
                temp1 = self.build_input(sentence)
                temp2 = self.build_input(sentences2[count])
                inputs = temp1[0] + temp2[0]
                labels = temp1[1] + temp2[1]
                for id in temp1[0]:
                    inputs += [id] * len(temp2[0])
                    labels += temp2[0]
                for id in temp2[0]:
                    inputs += [id] * len(temp1[0])
                    labels += temp1[0]

                times = int(math.ceil(float(len(inputs)) / self.batch_size))

                temp_loss = 0
                for t in range(times):
                    start = t * self.batch_size
                    end = min((t + 1) * self.batch_size, len(inputs))
                    batch_inputs = inputs[start:end]
                    batch_labels = labels[start:end]

                    batch_labels = np.reshape(batch_labels, [len(batch_labels), 1])

                    feed_dict = {self.inputs: batch_inputs,
                                 self.labels: batch_labels}

                    _, loss, summary = self.sess.run([self.optimizer, self.loss, self.merged], feed_dict=feed_dict)
                    self.writer.add_summary(summary, global_step=(num * len(sentences1) + count))
                    temp_loss += loss

                losses += temp_loss / times
                if (count + 1) % 100 == 0:
                    end_time = time.time()
                    print '--- epoch: %s --- training %.3f --- loss: %.3f --- cost: %.3f s ---' \
                          % (num, float(count + 1) / len(sentences1), losses / 100, start_time - end_time)
                    losses = 0
                    start_time = end_time
        self.saver.save(self.sess, '../save/' + self.name + self.time_stamp)

    def save(self, path):
        word_embeddings = self.sess.run(self.word_embeddings)
        temp = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))
        f = open(path, 'w')
        f.write(str(len(self.vocabulary)) + ' ' + str(self.size) + '\n')
        for i, embedding in enumerate(word_embeddings):
            f.write(temp[i] + ' ' + ' '.join(map(str, embedding)) + '\n')
        f.close()

    def init_op(self):
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter('../log/' + self.name + self.time_stamp, self.sess.graph)
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.sess.run(self.init)



    def build_network(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.int32, [None])
            self.labels = tf.placeholder(tf.int32, [None, 1])
            self.word_embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.size], -1.0, 1.0))
            self.weight = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.size], stddev=1.0 / math.sqrt(self.size)))
            self.bias = tf.Variable(tf.zeros([self.vocabulary_size]))

            embed = tf.nn.embedding_lookup(self.word_embeddings, self.inputs)

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(self.weight, self.bias, self.labels, embed, self.num_sample, self.vocabulary_size))

            tf.summary.scalar('loss', self.loss)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.init_op()

