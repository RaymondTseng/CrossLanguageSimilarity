

import csv
import numpy as np
import string
import random
from nltk.tokenize import WordPunctTokenizer
import time
import word2vec
import pandas as pd

def get_id(word):
    word = word.lower()
    return word2idx_.get(word, word2idx_['<unk>'])
    # if word in word2idx_:
    #     return word2idx_[word]
    # else:
    #     return word2idx_['<unk>']

def word2id(sentences, word2idx, seq_length):
    idx = []
    global word2idx_
    word2idx_ = word2idx
    for sentence in sentences:
        words = WordPunctTokenizer().tokenize(sentence)
        if len(words) < seq_length:
            for _ in range(len(words), seq_length):
                words.append('<unk>')
        elif len(words) > seq_length:
            words = words[:seq_length]
        id = map(get_id, words)
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

# en = []
# es = []
# f = open('/home/raymond/Downloads/es-en/europarl-v7.es-en.en')
# for line in f.readlines():
#     es.append(line.strip())
# f.close()
#
# f = open('/home/raymond/Downloads/es-en/europarl-v7.es-en.es')
# for line in f.readlines():
#     en.append(line.strip())
# f.close()
# f = open('/home/raymond/Downloads/es-en/europarl-v7.es-en.temp', 'w')
#
# for i, line in enumerate(en[:20000]):
#     if line and es[i]:
#         f.write(line + '|||' + es[i] + '|||1\n')
# ten = en[20000:40000]
# tes = es[20000:40000]
# import random
# random.shuffle(ten)
# random.shuffle(tes)
# for i, line in enumerate(ten):
#     if line and es[i]:
#         f.write(line + '|||' + tes[i] + '|||0\n')
# f.close()

























