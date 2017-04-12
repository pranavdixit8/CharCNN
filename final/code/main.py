# models - RNN, LSTM, GRU, BOW
# 
#
#
#
#
#
#from __future__ import print_function

import os
import sys
import string
from datetime import datetime
import cPickle

from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,Merge
from keras.datasets import imdb


import numpy as np
from Data import *

#print twitter_samples.fileids();

## GLOBAL PARAMETERS
maxlen = 80
batch_size = 32
epochs = 50

# if train is 0, load_saved_model_name is loaded
train = 1
test = 1
predict = 1
save_model = 0
load_saved_model_name = None
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1



num_lables = None



## LOAD AND PREPROCESS DATA

if train == 1:
	# arg1 - data_source - 'nltk'/'semeval'
	# arg2 - w2v_source - 'data'/'google'
	data = Data('nltk', 'data')

	if save_model == 1:
		outputdir = os.path.join(os.path.dirname(os.getcwd()),'models')
		now = str(datetime.now()).replace(':', '-')
		fname_out = 'Data-{}.pickle'.format(now)
		full_name = os.path.join(outputdir,fname_out)
		with open(full_name, 'wb') as fout:
			cPickle.dump(data, fout, -1)
		print "Saved model to {}".format(full_name)
		#print "Saved trained Data model to {}".format(full_name)
	else:
		print 'Not saving Data model' 

else:
	print "Loading Data model from {}".format(load_saved_model_name)
	data = cPickle.load(load_saved_model_name)

num_lables = data.num_lables


x_words,x_train,y_train = data.data,data.vec_data,data.y
total_in = len(y_train)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)


print total_in
x_val = x_train[int(train_ratio*total_in):int((train_ratio+val_ratio)*total_in)]
y_val = y_train[int(train_ratio*total_in):int((train_ratio+val_ratio)*total_in)]
x_test = x_train[int((train_ratio+val_ratio)*total_in):]
y_test = y_train[int((train_ratio+val_ratio)*total_in):]
x_train = x_train[:int(train_ratio*total_in)]
y_train = y_train[:int(train_ratio*total_in)]

def fork (model, n=2):
	forks = []
	for i in range(n):
		f = Sequential()
		f.add (model)
		forks.append(f)
	return forks

if train == 1:
	## TRAIN MODEL

	
   
	print('Build model...')
	left  = Sequential()
	#model.add(Embedding(max_features, 128))
	left.add(LSTM(128, input_shape=(80,100),dropout=0.2, return_sequences=True,recurrent_dropout=0.2))
	
	right = Sequential()
	right.add(LSTM(128, input_shape=(80,100),dropout=0.2, return_sequences=True,recurrent_dropout=0.2, go_backwards = True))
	
	model = Sequential()
	model.add(Merge([left,right],mode = 'concat'))
	#model.merge.average(left,right)
	
	left2, right2 = fork(model)
	
	#left2  = Sequential()
	#model.add(Embedding(max_features, 128))
	left2.add(LSTM(64, input_shape=(80,128),dropout=0.2, recurrent_dropout=0.2))
	
	#right2 = Sequential()
	right2.add(LSTM(64, input_shape=(80,128),dropout=0.2, recurrent_dropout=0.2, go_backwards = True))
	
	model = Sequential()
	model.add(Merge([left2,right2],mode = 'concat'))
	#model.merge.average(left2,right2)
	
	
	model.add(Dense(1, activation='sigmoid'))

	print model.summary()


	# try using different optimizers and different optimizer configs
	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	print('Train...')
	model.fit([x_train,x_train], y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          validation_data=([x_val,x_val], y_val))
	score, acc = model.evaluate([x_test,x_test], y_test, batch_size=batch_size)

	# evaluate loaded model on test data
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	#score = model.evaluate(x_train, y_train, verbose=0)
	#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))



	if save_model == 1:
		# serialize model to JSON
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
		    json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("model.h5")
		print("Saved model to disk")


else:
	# later...
	 
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("model.h5")
	print("Loaded model from disk")



## TEST MODEL

if predict == 1:
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
	result = model.predict([x_train,x_train])



	result = zip(result, x_words)

	for x,y in result:
	    print (y)
	    print (x)


