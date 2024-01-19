import os
import pandas as pd
import string
from pyvi import ViTokenizer
from gensim.models import Word2Vec

# path data
filename = 'datatrain.txt'
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, filename)

def read_data(path):
    traindata = []
    with open(path, 'r', encoding='utf-8') as file:
        sents = file.readlines()
        for sent in sents:
            traindata.append(sent.strip().split())
    return traindata

if __name__ == '__main__':
    train_data = read_data(file_path)
    model = Word2Vec(train_data, vector_size=100, window=5, min_count=1, workers=4)
    model.wv.save("./model/word2vec_skipgram.model")
