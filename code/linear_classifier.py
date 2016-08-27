import argparse
from ipdb import set_trace
import my_utils as ut
import numpy as np
from sklearn.svm import LinearSVC 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from my_utils.evaluation import accuracy
from my_utils.evaluation import AvgFmes
import codecs
from scipy.sparse import hstack
import sys

def load_lexicon(path, sep='\t', lex_low_bound=0, lex_up_bound=0):
	#we might want to filter out "neutral words"	
	with open(path) as fid:	
		lex =  {wrd: float(scr) for wrd, scr in (line.split(sep) for line in fid) if float(scr) < lex_low_bound or float(scr) > lex_up_bound}
	return lex

def lex_features(msg,lex):

	word_scores = map(lambda x:lex[x] if x in lex else 0, msg.split())
	mu = np.mean(word_scores)
	std_dev = np.std(word_scores)
	sum_score = sum(word_scores)
	max_score = max(word_scores)
	min_score = min(word_scores)
	#remove zeros
	word_scores_nozeros = filter(lambda x:x!=0, word_scores)
	if len(word_scores_nozeros)>0:
		last_score = word_scores_nozeros[-1]
	else:
		last_score = 0

	# print msg
	# print "mu ", mu
	# print "std_dev ", std_dev
	# print "sum ", sum_score
	# print "max ", max_score
	# print "min ", min_score
	# print "last ", last_score	

	return [mu,std_dev,sum_score,max_score,min_score,last_score]	

def get_parser():
    parser = argparse.ArgumentParser(description="Linear Classifier")
    
    #Basic Input
    parser.add_argument('-tr', type=str, required=True,
                        help='train file')

    parser.add_argument('-ts', type=str, required=True, nargs='+',
                        help='test file(s)')

    return parser

if __name__=="__main__":

	parser = get_parser()
	args = parser.parse_args()    

	#1. read files
	train_file = args.tr
	msgs_train = []
	labels_train = []
	print "training classifier"
	with open(train_file,"r") as fid:
		for l in fid:
			# print l
			splt = l.split("\t")
			labels_train.append(splt[0])
			msgs_train.append(splt[1])

	#2. build vocabulary
	wrd2idx = ut.word_2_idx(msgs_train)
	lbl2idx = ut.word_2_idx(labels_train)

	#3. get labels
	Y_train = np.array([lbl2idx[l] for l in labels_train])	

	# BOW features
	X_train = np.zeros((len(msgs_train),len(wrd2idx)))
	for i,m in enumerate(msgs_train):
		idx = [wrd2idx[w] for w in m.split() if w in wrd2idx]
		X_train[i,idx]=1

	#train
	clf = LinearSVC()
	clf.fit(X_train,Y_train)
	print "testing"
	#evaluate 
	for test_file in args.ts:		
		msgs_test = []
		labels_test = []
		with open(test_file) as fid:
			for l in fid:
				splt = l.split("\t")
				labels_test.append(splt[0])
				msgs_test.append(splt[1])
		Y_test  = np.array([lbl2idx[l] for l in labels_test])
		X_test = np.zeros((len(msgs_test),len(wrd2idx)))
		for i,m in enumerate(msgs_test):
			idx = [wrd2idx[w] for w in m.split() if w in wrd2idx]
			X_test[i,idx]=1
		y_hat = clf.predict(X_test)
		f1_avg_method = "binary" if len(set(labels_test))==2 else "macro"
		avgF1 = f1_score(Y_test, y_hat,average=f1_avg_method)
		acc   = accuracy(Y_test, y_hat)				
		print "%s ** acc: %.3f | F1: %.3f " % (test_file, acc, avgF1)


			
	

	
