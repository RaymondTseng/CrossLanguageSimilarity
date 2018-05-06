# -*- coding:utf-8 -*-
import random
from nltk.tokenize import WordPunctTokenizer
import nltk
import numpy as np
import re
from PIL import Image
punc = u"[\s+\.\!\/_,\-\?$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+"

nltk_pos_set = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'MNPS', 'NNS', 'PDT',
                'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.']

def get_id(word):
    word = word.strip().lower()
    return word2idx_.get(word, word2idx_['<0>'])


def word2id(sentences, word2idx, seq_length):
    idx = []
    all_length = []
    global word2idx_
    word2idx_ = word2idx
    for sentence in sentences:
        try:
            sentence = sentence.strip().decode('utf-8')
            sentence = re.sub(punc, u' ', sentence).strip()
            words = WordPunctTokenizer().tokenize(sentence)
        except:
            print(sentence)
        if len(words) < seq_length:
            all_length.append(len(words))
            for _ in range(len(words), seq_length):
                words.append('<0>')
        elif len(words) > seq_length:
            words = words[:seq_length]
            all_length.append(seq_length)
        else:
            all_length.append(seq_length)
        id = list(map(get_id, words))
        idx.append(id)
    return np.array(idx), np.array(all_length)

def tag2id(all_tags, tag2idx, seq_length):
    idx = []
    global word2idx_
    for tags in all_tags:
        if len(tags) < seq_length:
            for _ in range(len(tags), seq_length):
                tags.append('<0>')
        elif len(tags) > seq_length:
            tags = tags[:seq_length]
        id = []
        for tag in tags:
            id.append(tag2idx.get(tag, tag2idx['<unk>']))
        idx.append(id)

    return np.array(idx)

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
    index_set = set()
    for _ in range(batch_size):
        i = random.randint(0, len(sources) - 1)
        while i in index_set:
            i = random.randint(0, len(sources) - 1)
        index_set.add(i)
        sources_batch.append(sources[i])
        targets_batch.append(targets[i])
        scores_batch.append(scores[i])
    return sources_batch, targets_batch, scores_batch

def random_batch_with_handcraft(sources, targets, scores, source_features, target_features, batch_size):
    sources_batch = []
    targets_batch = []
    scores_batch = []
    source_features_batch = []
    target_features_batch = []
    for _ in range(batch_size):
        i = random.randint(0, len(sources) - 1)
        sources_batch.append(sources[i])
        targets_batch.append(targets[i])
        scores_batch.append(scores[i])
        source_features_batch.append(source_features[i])
        target_features_batch.append(target_features[i])
    return sources_batch, targets_batch, scores_batch, source_features_batch, target_features_batch

def get_handcraft_features(source, target, seq_length):
    source = WordPunctTokenizer().tokenize(source)
    target = WordPunctTokenizer().tokenize(target)
    source_features = np.zeros([seq_length, 2 + len(nltk_pos_set)], dtype=np.float32)
    target_features = np.zeros([seq_length, 2 + len(nltk_pos_set)], dtype=np.float32)
    source_tags = nltk.pos_tag(source)
    target_tags = nltk.pos_tag(target)
    for i in range(min(len(source), seq_length)):
        word_tag = source_tags[i]
        word = word_tag[0]
        tag = word_tag[1]
        if word in target:
            source_features[i][0] = 1.
            if word.isdigit():
                source_features[i][1] = 1.
        try:
            index = nltk_pos_set.index(tag)
        except:
            index = len(nltk_pos_set) - 1
        source_features[i][index + 2] = 1.
    for i in range(min(len(target), seq_length)):
        word_tag = target_tags[i]
        word = word_tag[0]
        tag = word_tag[1]
        if word in target:
            target_features[i][0] = 1.
            if word.isdigit():
                target_features[i][1] = 1.
        try:
            index = nltk_pos_set.index(tag)
        except:
            index = len(nltk_pos_set) - 1
        target_features[i][index + 2] = 1.
    return source_features, target_features

def get_all_handcraft_features(sources, targets, seq_length):
    all_source_features = []
    all_target_features = []
    for i, source in enumerate(sources):
        source_features, target_features = get_handcraft_features(source, targets[i], seq_length)
        all_source_features.append(source_features)
        all_target_features.append(target_features)
    return np.array(all_source_features), np.array(all_target_features)

def build_porbs(scores, class_num):
    probs = []
    for score in scores:
        score_floor = int(np.floor(score))
        prob = np.zeros(class_num)
        prob[score_floor] = score_floor - score + 1
        if score_floor + 1 < class_num:
            prob[score_floor + 1] = score - score_floor
        prob = np.clip(prob, 1e-6, 1.)
        probs.append(prob)
    return np.array(probs)

def normalize_probs(scores):
    return [score / 5. for score in scores]

def pearson(x1, x2):
    mid1 = np.mean(x1 * x2) - \
           np.mean(x1) * np.mean(x2)

    mid2 = np.sqrt(np.mean(np.square(x1)) - np.square(np.mean(x1))) * \
           np.sqrt(np.mean(np.square(x2)) - np.square(np.mean(x2)))

    pearson = mid1 / mid2

    return pearson

def pos_tag(all_words):
    all_tags = []
    for words in all_words:
        tokens = nltk.pos_tag(words)
        tags = [token[1] for token in tokens]
        all_tags.append(tags)
    return all_tags

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def weight2img(weights):
    weights = sigmoid(weights) * 255
    img = Image.fromarray(weights)
    img.show()

# lines = []
#
# f = open('/media/raymond/CE687D43687D2B7B/data/paragram_300_sl999.txt', 'r')
# for i, line in enumerate(f.readlines()):
#
#     parts = line.decode('utf-8').strip().split(' ')
#     if len(parts) == 301:
#         lines.append(line)
#
# f.close()
# f = open('/media/raymond/CE687D43687D2B7B/data/paragram_300_sl999_1.txt', 'w')
# f.write(str(len(lines)) + ' 300\n')
# for line in lines:
#     f.write(line)
# f.close()
