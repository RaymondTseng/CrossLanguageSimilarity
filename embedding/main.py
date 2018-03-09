# -*- coding:utf-8 -*-
from model import CrossLingualWord2Vec
from nltk.tokenize import WordPunctTokenizer


en_sentences = []
es_sentences = []
print 'loading data...'
en = open('/home/raymond/Downloads/es-en/europarl-v7.es-en.en', 'r').readlines()
es = open('/home/raymond/Downloads/es-en/europarl-v7.es-en.es', 'r').readlines()
for i, line1 in enumerate(en):
    line1 = line1.strip()
    line2 = es[i].strip()
    if line1 and line2:
        words1 = WordPunctTokenizer().tokenize(line1)
        words2 = WordPunctTokenizer().tokenize(line2)
        if len(words1) > 1 and len(words2) > 1:
            en_sentences.append([word.lower() for word in words1])
            es_sentences.append([word.lower() for word in words2])
del en, es
print 'build model...'

model = CrossLingualWord2Vec(en_sentences, es_sentences)
model.save('model.en-es')

