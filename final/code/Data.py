import random
from Preprocess import *
from Word2vec import *
import numpy as np

class Data:
	def __init__(self, data_source, w2v_source):
		self.data_source = data_source
		self.num_lables = None
		self.w2v_source = w2v_source
		self.load_data()


	def load_data(self):
		#self.data = None
		if (self.data_source == 'semeval'):
			self.load_semeval()
			self.num_lables = 3
		elif (self.data_source == 'nltk'):
			self.load_nltk()
			self.num_lables = 2
		self.convert2vec()
		self.shuffle()



	# load and preprocess(tokenize)
	def load_semeval(self):
		print 'Loading Semeval data'
		print '###TBD###'

	# load and preprocess(tokenize)
	def load_nltk(self):
		print 'Loading nltk data'
		from nltk.corpus import twitter_samples

		#strings = twitter_samples.strings('tweets.20150430-223406.json');
		strings = twitter_samples.strings('negative_tweets.json');
		temp = len(strings)
		self.y = [0]*temp
		strings = strings + twitter_samples.strings('positive_tweets.json');
		self.y = self.y + ([1]*(len(strings) - temp))
		self.sentences = strings
		#tokenize
		print 'Tokenize'
		self.preprocesser = Preprocess(strings, 'tweet')
		self.data = self.preprocesser.op_data
		
		#debug
		#f1 = open('temp.txt', 'w+')
		#for x in self.data:
		#	print >> f1, x

	def convert2vec(self):

		#word2vec
		print 'word2vec'
		self.w2v_model = Word2vec(self.data, self.w2v_source)
		self.vec_data = self.w2v_model.op_data
		#self.vec_data = Word2vec(self.data, 'google').op_data

		# chenge to numpy
		# check if it is needed - sentences are of diff length
		'''print >> f1,self.vec_data[0]
		na = np.array(self.vec_data)
		print >> f1,na[0]
		print '##na.shape'
		print na.shape
		'''

	def shuffle(self):
		# append y
		combined = list(zip(self.sentences, self.data, self.vec_data, self.y))
		# suffle
		random.shuffle(combined)
		# seperate
		self.sentences[:], self.data[:], self.vec_data[:], self.y[:] = zip(*combined)

		'''for i in range(5):
			print >> f1, self.data[i]
			print >> f1, self.vec_data[i]
			print >> f1, self.y[i]'''

	def preprocess_sent(self, sent):
		toks = self.preprocesser.preprocess_sent(sent)
		f1 = open('temp.txt', 'w+')
		print >>f1, self.w2v_model.w2v_sent(toks)
		return self.w2v_model.w2v_sent(toks)
		

