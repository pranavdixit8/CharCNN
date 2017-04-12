# twitter.tokenized uses casual.py tokenizer
#from nltk.tokenize import casual
# This is same as above
from nltk.tokenize import TweetTokenizer

#from nltk.tokenize import sent_tokenize, word_tokenize

class Preprocess:
	def __init__(self, ip_data, pp_type):
		self.tokenizer = None
		self.op_data = None
		print 'Preprocessing data'
		self.process(ip_data, pp_type)

	def process(self, ip_data, pp_type):
		if(pp_type == 'tweet'):
			self.tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True)
		elif(pp_type == '???'):
			# TBD
			print '###TBD###'

		self.op_data = []
		for sent in ip_data:
			self.op_data.append(self.tokenizer.tokenize(sent))


	def preprocess_sent(self, sent):
		return self.tokenizer.tokenize(sent)
