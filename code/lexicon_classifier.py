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

def load_lexicon(path, sep='\t', lex_thresh=0.25):
	#we might want to filter out "neutral words"	
	with open(path) as fid:	
		lex =  { wrd: float(scr) for wrd, scr in (line.split('\t') for line in fid) if float(scr) < -lex_thresh or float(scr) > lex_thresh }
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
    parser.add_argument('-lex', type=str, required=True,
                        help='lexicon file')

    parser.add_argument('-ts', type=str, required=True, nargs='+',
                        help='test file(s)')

    return parser

if __name__=="__main__":
	DECISION_BOUNDARY = 0
	parser = get_parser()
	args = parser.parse_args()  
	lex = load_lexicon(args.lex)
	print "testing"
	#evaluate 
	for test_file in args.ts:		
		msgs_test = []
		labels_test = []
		scores = []
		with open(test_file) as fid:
			for l in fid:
				splt = l.split("\t")
				labels_test.append(splt[0])
				msgs_test.append(splt[1])
				word_scores = map(lambda x:lex[x] if x in lex else 0, splt[1].split())
				word_scores = filter(lambda x:x!=0, word_scores)
				if len(word_scores)>0:
					msg_score = np.mean(word_scores)				
				else:				
					msg_score = 0		
				scores.append(np.mean(msg_score))
		lbl2idx = ut.word_2_idx(labels_test)
		Y_test  = np.array([lbl2idx[l] for l in labels_test])	
		y_hat   = map(lambda x:lbl2idx["positive"] if x >= DECISION_BOUNDARY else lbl2idx["negative"], scores) 
		# set_trace()		
		avgF1 = f1_score(Y_test, y_hat,average="binary")
		acc   = accuracy_score(Y_test, y_hat)				
		print "%s ** acc: %.3f | F1: %.3f " % (test_file, acc, avgF1)


			
	

	
