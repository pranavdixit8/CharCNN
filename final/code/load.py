from __future__ import print_function

import os
import sys
import string
from datetime import datetime
import cPickle

from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


import numpy as np
from Data import *

# #print twitter_samples.fileids();
# 
# ## GLOBAL PARAMETERS
# maxlen = 80
# batch_size = 32
# train = 1
# test = 1
# 
# 
# num_lables = None
# 
# ## LOAD AND PREPROCESS DATA
# # arg1 - data_source - 'nltk'/'semeval'
# # arg2 - w2v_source - 'data'/'google'
data = Data('nltk', 'data')
# 
# x_words,x_data,x_train,y_train = data.words,data.data,data.vec_data,data.y
# 
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# #x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# #print('x_test shape:', x_test.shape)
# 
# num_lables = data.num_lables

# max_features = 20000
# maxlen = 80  # cut texts after this number of words (among top max_features most common words)
# batch_size = 32
# 
# print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
# 
# print(x_test[0])
# print(x_test[1])
# 
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# 
# print(x_test[0])
# print(x_test[1])
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data9
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
maxlen =80
x_train = []
x_words = []
x_words.append('This is good')
x_words.append('This is really good')
x_words.append('This is not really good')
x_words.append('This is too bad')
for sent in x_words:
    x_train.append(data.preprocess_sent(sent))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)
#result = loaded_model.predict(x_train[:10])
result = loaded_model.predict(x_train)



result = zip(result, x_words)

for x,y in result:
    print (y)
    print (x)
    
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# now = str(datetime.now()).replace(':', '-')
# fname_out = 'fulltrain-{}.pickle'.format(now)
# full_name = os.path.join(outputdir,fname_out)
# 
# with open(full_name, 'wb') as fout:
#     cPickle.dump(model, fout, -1)
# print ("Saved model to {}".format(full_name))

