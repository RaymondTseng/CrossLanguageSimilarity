import random
from nltk.tokenize import WordPunctTokenizer


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
                words.append('<unki>')
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

def get_handcraft_features(sources, targets, seq_length):
    for i in range(len(sources)):
        pass

import nltk
text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)



























