import pandas as pd
import csv
import word2vec
import numpy as np
import random
import os

def load_cross_lang_sentence_data(path):
    source_sentences = []
    target_sentences = []
    scores = []
    f = open(path, 'r')
    lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        source_sentences.append(temp[0])
        target_sentences.append(temp[1])
        scores.append(float(temp[2]))
    f.close()
    return source_sentences, target_sentences, scores

def load_sick_data(path):
    df_sick = pd.read_csv(path, sep="\t", usecols=[1, 2, 4], names=['s1', 's2', 'score'],
                          dtype={'s1': object, 's2': object, 'score': object})
    df_sick = df_sick.drop([0])
    sources = df_sick.s1.values
    targets = df_sick.s2.values
    scores = np.asarray(map(float, df_sick.score.values), dtype=np.float32)
    return sources, targets, scores

def load_sts_data(path):
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

    return sources, targets, scores

def load_embedding(path):
    wv = word2vec.load(path)
    vocab = wv.vocab
    word2idx = {}
    for i in range(1, len(vocab) + 1):
        word2idx[vocab[i-1]] = i
    word2idx['<unk>'] = 0
    word_embedding = wv.vectors
    word_mean = np.mean(word_embedding, axis=0)
    word_embedding = np.vstack([word_mean, word_embedding])
    return word2idx, word_embedding


