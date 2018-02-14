

import csv
import numpy as np
import string
import random
import word2vec
import pandas as pd

train_path = '/home/raymond/Downloads/data/sts-train.csv'
embedding_path = '/home/raymond/Downloads/data/glove.6B.50d.txt'
word2idx_path = 'word2idx.txt'

def load_csv(path):
    csv_reader = csv.reader(open(path))
    scores = []
    sources = []
    targets = []
    for row in csv_reader:
        temp = row[0].split('\t')
        if len(temp) >= 7:
            scores.append(float(temp[4]))
            sources.append(temp[5])
            targets.append(temp[6])

    return scores, sources, targets

def load_embedding(path):
    wv = word2vec.load(path)
    vocab = wv.vocab
    word2idx = pd.Series(range(1, len(vocab) + 1), index=vocab)
    word2idx['<unk>'] = 0
    word_embedding = wv.vectors
    word_mean = np.mean(word_embedding, axis=0)
    word_embedding = np.vstack([word_mean, word_embedding])
    return word2idx, word_embedding

def word2id(sentences, word2idx, seq_length):
    idx = []
    for sentence in sentences:
        words = sentence.translate(None, string.punctuation).split(' ')
        words = [word.lower() for word in words]
        id = [word2idx.get(word, word2idx['<unk>']) for word in words]
        if len(id) < seq_length:
            for _ in range(len(id), seq_length):
                id.append(word2idx['<unk>'])
        elif len(id) > seq_length:
            id = id[:seq_length]
        idx.append(id)
    return idx

def load_word2idx(path):
    word2idx = {}
    f = open(path, 'r')
    for line in f.readlines():
        temp = line.split(',|')
        word2idx[temp[0]] = int(temp[1])
    f.close()
    word2idx['UNK'] = len(word2idx)
    return word2idx

def random_batch(sources, targets, scores, batch_size):
    sources_batch = []
    targets_batch = []
    scores_batch = []
    for _ in range(batch_size):
        i = random.randint(0, len(sources) - 1)
        sources_batch.append(sources[i])
        targets_batch.append(targets[i])
        scores_batch.append(scores[i])
    return sources_batch, targets_batch, scores_batch

















