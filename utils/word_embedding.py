from gensim.models import word2vec
import os
from nltk.tokenize import WordPunctTokenizer
import logging





# en_sentences = []
#
# print 'loading data...'
# en = open('/home/raymond/Downloads/es-en/europarl-v7.es-en.en', 'r').readlines()
#
# for i, line1 in enumerate(en):
#     line1 = line1.strip()
#
#     if line1:
#         words1 = WordPunctTokenizer().tokenize(line1)
#
#         if len(words1) > 1:
#             en_sentences.append([word.lower() for word in words1])
#
# logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
# model = word2vec.Word2Vec(en_sentences, size=50)
# model.save('/home/raymond/Downloads/es-en/model.en.50')

model = word2vec.Word2Vec.load('/home/raymond/Downloads/data/spanish.news.50d.model')
model.train()
f = open('/home/raymond/Downloads/data/spanish.news.50d.txt', 'w')
f.write(str(len(model.wv.index2word)) + ' ' + str(model.wv.vector_size) + '\n')
for i, word in enumerate(model.wv.index2word):
    line = word + ' ' + ' '.join(map(str, model.wv.syn0[i]))
    f.write(line.encode('utf-8') + '\n')
f.close()
print(model)