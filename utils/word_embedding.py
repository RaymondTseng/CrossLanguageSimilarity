from gensim.models import word2vec
import os
from nltk.tokenize import WordPunctTokenizer
import nltk.data
import logging

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

news_path = '/home/raymond/Downloads/spanish_news/'
others_path = '/home/raymond/Downloads/spanish_others/'
save_path = '/home/raymond/Downloads/data/spanish.news.50d.model'

# all_words = []
#
# for (root, dirs, files) in os.walk(news_path):
#     for file_name in files:
#         file_name = os.path.join(root, file_name)
#         f = open(file_name, 'r')
#         for line in f.readlines():
#             line = line.strip().decode('utf-8')
#             sentences = tokenizer.tokenize(line)
#             for sentence in sentences:
#                 words = WordPunctTokenizer().tokenize(sentence)
#                 all_words.append(words)
#         f.close()
# for (root, dirs, files) in os.walk(others_path):
#     for file_name in files:
#         file_name = os.path.join(root, file_name)
#         f = open(file_name, 'r')
#         for line in f.readlines():
#             line = line.strip().decode('utf-8')
#             sentences = tokenizer.tokenize(line)
#             for sentence in sentences:
#                 words = WordPunctTokenizer().tokenize(sentence)
#                 all_words.append(words)
#         f.close()
#
# logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
# model = word2vec.Word2Vec(all_words, size=50)
# model.save(save_path)

model = word2vec.Word2Vec.load(save_path)

f = open('/home/raymond/Downloads/data/spanish.news.50d.txt', 'w')
f.write(str(len(model.wv.index2word)) + ' ' + str(model.wv.vector_size) + '\n')
for i, word in enumerate(model.wv.index2word):
    line = word + ' ' + ' '.join(map(str, model.wv.syn0[i]))
    f.write(line.encode('utf-8') + '\n')
f.close()
print(model)