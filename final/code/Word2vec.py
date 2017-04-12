import gensim, logging
#import cPickle
import numpy as np
import os
import sys
from datetime import datetime

# Train word2vec and convert data to vector		
class Word2vec:
	def __init__(self, in_data, model_type):
		self.in_data = in_data
		self.op_data = None
		self.model_type = model_type
		self.get_model()


	def get_model(self):
		if (self.model_type == 'google'):
			self.load_and_train_google()
		elif (self.model_type == 'data'):
			# use data to train
			self.train_data()
		self.compute_vec()


	def load_and_train_google(self):
		# https://github.com/chrisjmccormick/inspect_word2vec/blob/master/inspect_google_word2vec.py
		# Load Google's pre-trained Word2Vec model.
		print os.path.exists('../models/GoogleNews-vectors-negative300.bin.gz')
		#self.model = gensim.models.Word2Vec.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
		self.model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
		self.vec_size = 300
		'''import inspect
		print self.model.__class__.__name__
		print inspect.getmembers(self.model.__class__, lambda a:not(inspect.isroutine(a)))
		sys.exit()'''




		# Does the model include stop words?
		print("Does it include the stop words like \'a\', \'and\', \'the\'? %d %d %d" % ('a' in self.model.vocab, 'and' in self.model.vocab, 'the' in self.model.vocab))

		# Retrieve the entire list of "words" from the Google Word2Vec model, and write
		# these out to text files so we can peruse them.
		vocab = self.model.vocab.keys()

		fileNum = 1

		wordsInVocab = len(vocab)

		print 'wordsInVocab'
		print wordsInVocab


		# tune google data with in_data
		# TBD
		#sentences = self.in_data
		#self.model.train(sentences, min_count = 1, workers = 2)


		# truncate model to current vocab
		# TBD - check if needed


		# save tuned model
		'''outputdir = os.path.join(os.path.dirname(os.getcwd()),'models')
		now = str(datetime.now()).replace(':', '-')
		fname_out = 'fulltrain-word2vec-google-{}.pickle'.format(now)
		full_name = os.path.join(outputdir,fname_out)

		self.model.save(full_name)
		#with open(full_name, 'wb') as fout:
		#    cPickle.dump(model, fout, -1)

		print "Saved model to {}".format(full_name)'''



	def train_data(self):
		# https://rare-technologies.com/word2vec-tutorial/
		# http://radimrehurek.com/gensim/models/word2vec.html
		# TBD - check
		sentences = self.in_data
		#model = gensim.models.Word2Vec(sentences, min_count=1)
		# workers is number of processes
		# min_count is ignore words that occur less than min_count #times
		self.vec_size = 100
		self.model = gensim.models.Word2Vec(sentences, size = self.vec_size, min_count = 1, workers = 2)
		
		outputdir = os.path.join(os.path.dirname(os.getcwd()),'models')
		now = str(datetime.now()).replace(':', '-')
		fname_out = 'fulltrain-word2vec-data-{}.pickle'.format(now)
		full_name = os.path.join(outputdir,fname_out)

		self.model.save(full_name)
		#with open(full_name, 'wb') as fout:
		#    cPickle.dump(model, fout, -1)

		print "Saved model to {}".format(full_name)


	def compute_vec(self):

		# to load saved model
		# self.model = Word2Vec.load(full_name)

		# example to get vector for a word
		#>>> self.model.wv['computer']  # numpy vector of a word
		#array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)


		self.op_data = []
		if self.model_type == 'data':
			for sent in self.in_data:
				sent_vec = []
				for word in sent:
					sent_vec.append(self.model.wv[word])
				self.op_data.append(sent_vec)
		elif self.model_type == 'google':
			for sent in self.in_data:
				sent_vec = []
				for word in sent:
					try:
						c = self.model.word_vec(word)
						sent_vec.append(c)
					except KeyError:
						print 'word' + word + 'is not in dictionary'
						#c = np.array([0]*self.vec_size)
				self.op_data.append(sent_vec)
				
		


	def w2v_sent(self, sent):
		vec = []
		if self.model_type == 'data':
			for word in sent:
				try:
					c = self.model.wv[word]
				except KeyError:
					print 'word' + word + 'is not in dictionary'
					c = np.array([0]*self.vec_size)
				vec.append(self.model.wv[word])
			return vec
		elif self.model_type == 'google':
			for word in sent:
				try:
					c = self.model.word_vec(word)
					vec.append(c)
				except KeyError:
					print 'word' + word + 'is not in dictionary'
					#c = np.array([0]*self.vec_size)
				
			return vec
		

