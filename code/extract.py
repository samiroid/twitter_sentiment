import argparse
from bs4 import BeautifulSoup
import codecs
from ipdb import set_trace
from my_utils import preprocess
from my_utils import word_2_idx
from random import shuffle
import sys

####
# Prepare the datasets
# 1. read and preprocess + tokenize
# 2. write files with format: label \t text \n
####

DATA_IN = "DATA/input/"
DATA_OUT = "DATA/txt/"

def read_omd(binary=False):
	# ---- OMD
	all_msgs = []
	for f in ["dev.xml","train.xml","test.xml"]:	
		msgs = []
		with open(DATA_IN+"omd/%s" % f) as fid:
			soup = BeautifulSoup(fid.read(),"xml")
			for item in soup.findAll('item'):				
				if (binary and item.attrs['label'] not in ["positive","negative"]) \
					        or item.attrs['label'] not in ["positive","neutral","negative"] : continue
				msg = item.find("content").text
				msg = preprocess(msg.decode("utf-8"))	
				ex = (item.attrs['label'],msg)
				msgs.append(ex)
			shuffle(msgs)
		all_msgs += msgs	
		fname = "omd_%s.txt" % f.replace(".xml","")
		with open(DATA_OUT+fname,"w") as fod:
			for ex in msgs:
				fod.write('\t'.join(ex)+"\n")

	return all_msgs

def read_hcr(binary=False):
	# ---- HCR
	all_msgs = []
	for f in ["dev.xml","train.xml","test.xml"]:	
		msgs = []
		with open(DATA_IN+"hcr/%s" % f) as fid:
			soup = BeautifulSoup(fid.read(),"xml")
			for item in soup.findAll('item'):				
				if (binary and item.attrs['label'] not in ["positive","negative"]) \
					        or item.attrs['label'] not in ["positive","neutral","negative"] : continue				
				msg = item.find("content").text
				msg = preprocess(msg.decode("utf-8"))	
				ex = (item.attrs['label'],msg)
				msgs.append(ex)
			shuffle(msgs)
		all_msgs += msgs	
		fname = "hcr_%s.txt" % f.replace(".xml","")
		with open(DATA_OUT+fname,"w") as fod:
			for ex in msgs:
				fod.write('\t'.join(ex)+"\n")

	return all_msgs

def read_semeval(binary=False):
	# ---- semeval
	all_msgs = []
	for fname in ["semeval_train_complete.txt","Twitter2013_raw.txt","Twitter2014_raw.txt","Twitter2015_raw.txt"]:	
		msgs = []
		with codecs.open(DATA_IN+"semeval/%s" % fname,"r","utf-8") as fid:
			for l in fid:
				spt = l.replace("\n","").split("\t")			
				label = spt[0].replace("\"","")
				if label == "objective-OR-neutral": 
					label = "neutral"	
				if (binary and label not in ["positive","negative"]) \
					        or label not in ["positive","neutral","negative"] : continue			
				tweet = spt[1]
				tweet = preprocess(tweet)				
				ex = (label,tweet)
				msgs.append(ex)
		shuffle(msgs)
		all_msgs += msgs			
		with codecs.open(DATA_OUT+fname.lower(),"w","utf-8") as fod:
			for ex in msgs:
				fod.write('\t'.join(ex)+"\n")
	
	return all_msgs

def get_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-bin', action="store_true", default=False,
                        help='if True, only positive and negatives will be used')
    return parser

if __name__ == "__main__":

	parser = get_parser()
	args = parser.parse_args()    

	if args.bin:
		print "Binary labels"
	instances = []
	instances += read_omd(args.bin)
	instances += read_hcr(args.bin)
	instances += read_semeval(args.bin)
	msgs = [inst[1] for inst in instances]
	wrd2idx = word_2_idx(msgs)
	with codecs.open("DATA/txt/vocab.txt","w","utf-8") as fid:
		for w in wrd2idx.keys():
			fid.write(w+"\n")

