# encoding:utf-8

import numpy as np
import tensorflow as tf
#from pretretment import get_file_name
from gensim.models import Word2Vec


event_txt_path = '../data/event_text/'
vector_path = '../data/word2vec/'

def get_embeddings(word_path, vec_path):
    model = Word2Vec.load(vec_path + 'word2vec.model')
    vector = model.wv

    with open(event_txt_path + 'E100.txt', 'r', encoding='utf8') as event:
        lines = event.readlines()
        n = len(lines)
        vec = np.zeros((n, 300), dtype=np.float32)
        for i in range(n):
            sents = lines[i].split()
            for word in sents:
                vec[i, ] += np.array(vector[word])


if __name__ == '__main__':
    get_embeddings(event_txt_path, vector_path)
