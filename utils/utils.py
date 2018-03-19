import random
from nltk.tokenize import WordPunctTokenizer
import nltk
import numpy as np
import string

nltk_pos_set = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'MNPS', 'NNS', 'PDT',
                'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.']

def get_id(word):
    word = word.lower()
    return word2idx_.get(word, word2idx_['<unk>'])


def word2id(sentences, word2idx, seq_length):
    idx = []
    global word2idx_
    word2idx_ = word2idx
    for sentence in sentences:
        try:
            words = WordPunctTokenizer().tokenize(sentence)
        except:
            print(sentence)
        if len(words) < seq_length:
            for _ in range(len(words), seq_length):
                words.append('<unk>')
        elif len(words) > seq_length:
            words = words[:seq_length]
        id = list(map(get_id, words))
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

# es = open('/home/raymond/Downloads/es-en/europarl-v7.es-en.en', 'r').readlines()
# en = open('/home/raymond/Downloads/es-en/europarl-v7.es-en.en', 'r').readlines()
# lines = []
# index_set = set()
#
# while (len(index_set) < 5000):
#     index = random.randint(0, len(es) - 1)
#     while (index in index_set):
#         index = random.randint(0, len(es) - 1)
#     words1 = es[index].strip().split(' ')
#     words2 = en[index].strip().split(' ')
#     if (2 < len(words1) < 45 and 2 < len(words2) < 45):
#         lines.append(es[index].strip() + '\t' + en[index].strip() + '\t1\n')
#         index_set.add(index)
#
# while (len(index_set) < 15000):
#     index1 = random.randint(0, len(es) - 1)
#     while (index1 in index_set):
#         index1 = random.randint(0, len(es) - 1)
#     index2 = random.randint(0, len(es) - 1)
#     while (index2 in index_set):
#         index2 = random.randint(0, len(es) - 1)
#     if index1 == index2:
#         continue
#     words1 = es[index1].strip().split(' ')
#     words2 = en[index2].strip().split(' ')
#     if (2 < len(words1) < 45 and 2 < len(words2) < 45):
#         lines.append(es[index1].strip() + '\t' + en[index2].strip() + '\t0\n')
#         index_set.add(index1)
#         index_set.add(index2)
# np.random.shuffle(lines)
#
# f = open('/home/raymond/Downloads/es-en/cls-train.txt', 'w')
# for line in lines[:8000]:
#     f.write(line)
# f.close()
# f = open('/home/raymond/Downloads/es-en/cls-dev.txt', 'w')
# for line in lines[8000:9000]:
#     f.write(line)
# f.close()
# f = open('/home/raymond/Downloads/es-en/cls-test.txt', 'w')
# for line in lines[9000:]:
#     f.write(line)
# f.close()
























