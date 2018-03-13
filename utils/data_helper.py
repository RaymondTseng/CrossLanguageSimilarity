import pandas as pd
import csv
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

def load_europarl_data(path):
    source_sentences = []
    target_sentences = []
    scores = []
    f = open(path, 'r')
    lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
        line = line.strip()
        temp = line.split('|||')
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
    csv_reader = csv.reader(open(path, encoding='UTF-8'))
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

def load_embedding(path, unk):
    word2idx = {}
    size = dimension = word_embeddings = None
    f = open(path, 'r', encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        temp = line.strip().split(' ')
        if i == 0:
            assert len(temp) == 2
            size = int(temp[0])
            dimension = int(temp[1])
            word_embeddings = np.zeros([size + 1, dimension], dtype=np.float64)
        else:
            assert len(temp) == dimension + 1
            word2idx[temp[0]] = i
            word_embeddings[i] = np.array(temp[1:], dtype=np.float64)
    word2idx['<0>'] = 0
    word_embeddings = np.vstack([np.zeros([dimension], dtype=np.float64), word_embeddings])
    if unk:
        word2idx['<unk>'] = len(word2idx)
        word_mean = np.mean(word_embeddings, axis=0, keepdims=False)
        word_embeddings = np.vstack([word_embeddings, word_mean])
    return word2idx, word_embeddings





