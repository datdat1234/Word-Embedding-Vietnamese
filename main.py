from gensim.models import KeyedVectors
import sys

model = KeyedVectors.load('./model/word2vec_skipgram.model')

arguments = sys.argv
script_name = arguments[0]
command_line_arguments = arguments[1:]
sims = model.most_similar(command_line_arguments[0], topn=5) 
print(sims)