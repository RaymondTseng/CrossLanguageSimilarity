

import csv
import numpy as np
import string
import random
import word2vec
import pandas as pd


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


















