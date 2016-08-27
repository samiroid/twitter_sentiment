import argparse
from bs4 import BeautifulSoup
from collections import Counter
import codecs
from ipdb import set_trace
from my_utils import preprocess
from my_utils import word_2_idx
from my_utils.embeddings import save_embeddings_txt
from random import shuffle
import sys

####
# Prepare all the datasets
# 1. read and preprocess + tokenize
# 2. shuffle and split (keeping class proportions)
# 3. write files with format: label [] \t text \n
# 4. extract the full vocabulary and the corresponding embedding matrix 
# 4.1 for words without an embedding randomly initialize
####

DATA_IN = "DATA/input/"
DATA_OUT = "DATA/txt/"

def shuffle_split(positives,negatives,neutrals,split_perc = 0.8):
	#shuffle
	shuffle(positives)
	shuffle(negatives)
	shuffle(neutrals)
	#split
	train = positives[:int(len(positives)*.8)] + negatives[:int(len(negatives)*.8)] + neutrals[:int(len(neutrals)*.8)]
	test = positives[int(len(positives)*.8):] + negatives[int(len(negatives)*.8):] + neutrals[int(len(neutrals)*.8):]
	#reshuffle
	shuffle(train)
	shuffle(test)

	return train, test

def get_omd(binary=False):
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

def get_hcr(binary=False):
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

def get_semeval(binary=False):
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


# def get_stance():
# 	tweets = []
# 	def process_stance(target_filter,out_file):		
# 		positives = []
# 		negatives = []
# 		neutrals  = []	
# 		for f in ["train","test","trial"]:	
# 			with open(DATA_IN+"stance/Stance2016_%s.txt" % f) as fid:
# 				for l in fid:
# 					spt = l.replace("\r\n","").split("\t")
# 					target, tweet, label = spt[1:]		
# 					tweet = preprocess(tweet)			
# 					if target!=target_filter:
# 						continue
# 					# ex = (target,label,tweet)
# 					ex = [label,tweet]
# 					if label == 'FAVOR':
# 						ex[0]='positive'
# 						positives.append(ex)
# 					elif label == 'AGAINST':
# 						ex[0]='negative'
# 						negatives.append(ex)
# 					elif label == 'NONE':
# 						ex[0]='neutral'
# 						neutrals.append(ex)
		
# 		train,test = shuffle_split(positives, negatives, neutrals)

# 		with open(DATA_OUT+"%s_train.txt" % out_file,"w") as fod:
# 			for ex in train:
# 				fod.write('\t'.join(ex)+"\n")

# 		with open(DATA_OUT+"%s_test.txt" % out_file,"w") as fod:
# 			for ex in test:
# 				fod.write('\t'.join(ex)+"\n")
# 		return train + test
# 	# --- STANCE 
# 	stance_ds = [('Atheism','SD-A'), ('Legalization of Abortion','SD-LA'), ('Feminist Movement','SD-FM'), ('Climate Change is a Real Concern','SD-CC'), ('Hillary Clinton','SD-HC')]

# 	for s in stance_ds:
# 		tweets += process_stance(s[0],s[1])

# 	return tweets 

# def get_semeval():	
# 	def process_semeval(dataset):
# 		positives = []
# 		negatives = []
# 		neutrals  = []
# 		with codecs.open(DATA_IN+"semeval/%s.txt" % dataset,"r","utf-8") as fid:
# 			for l in fid:
# 				spt = l.replace("\n","").split("\t")			
# 				label = spt[0].replace("\"","")
# 				if label == "objective-OR-neutral": 
# 					label = "neutral"						
# 				tweet = spt[1]
# 				tweet = preprocess(tweet)
# 				# tweet = tweet.encode("utf-8")
# 				# print label, tweet
# 				ex = (label,tweet)
# 				# set_trace()
# 				if label == 'positive':
# 					positives.append(ex)
# 				elif label == 'negative':
# 					negatives.append(ex)
# 				elif label == 'neutral':
# 					neutrals.append(ex)
# 				else:
# 					raise NotImplementedError
# 		train,test = shuffle_split(positives, negatives, neutrals)
# 		with codecs.open(DATA_OUT+"%s_train.txt" % dataset.upper(),"w","utf-8") as fod:
# 				for ex in train:
# 					fod.write(u'\t'.join(ex)+u"\n")

# 		with codecs.open(DATA_OUT+"%s_test.txt" % dataset.upper() ,"w","utf-8") as fod:
# 			for ex in test:
# 				fod.write(u'\t'.join(ex)+u"\n")

# 		return train + test
# 	tweets = []
# 	datasets = ["semeval_train_complete","Twitter2013_raw","Twitter2014_raw","Twitter2015_raw"]
# 	for d in datasets:
# 		tweets+= process_semeval(d)
	
# 	return tweets




# def get_duggan():
# 	cache = dict()
# 	positives = []
# 	negatives = []
# 	neutrals  = []	
# 	with open(DATA_IN+"RWA/RWA-duggan.txt","r") as fid:
# 		for l in fid:
# 			spt = l.split("\t")
# 			label = spt[0].split(",")[1]
# 			tweet = preprocess(spt[1])
# 			if tweet in cache: continue
# 			cache[tweet]=True
# 			# print label, tweet
# 			ex = [label,tweet]
# 			if label == 'justice':
# 				ex[0]='positive'
# 				positives.append(ex)
# 			elif label == 'nojustice':
# 				ex[0]='negative'
# 				negatives.append(ex)
# 			elif label == 'other':
# 				ex[0]='neutral'
# 				neutrals.append(ex)

# 	train,test = shuffle_split(positives, negatives, neutrals)

# 	with open(DATA_OUT+"RWA-MD_train.txt","w") as fod:
# 			for ex in train:
# 				fod.write('\t'.join(ex)+"\n")

# 	with open(DATA_OUT+"RWA-MD_test.txt","w") as fod:
# 		for ex in test:
# 			fod.write('\t'.join(ex)+"\n")

# 	return train + test

# def get_isis():
# 	cache = dict()
# 	positives = []
# 	negatives = []
# 	neutrals  = []	
# 	with open(DATA_IN+"RWA/RWA-isis.txt","r") as fid:
# 		for l in fid:
# 			spt = l.split("\t")
# 			label = spt[0].split(",")[1]
# 			tweet = preprocess(spt[1])
# 			if tweet in cache: continue
# 			cache[tweet]=True
# 			# print label
# 			ex = [label,tweet]
# 			if label == 'pro':
# 				ex[0]='positive'
# 				positives.append(ex)
# 			elif label == 'anti':
# 				ex[0]='negative'
# 				negatives.append(ex)
# 			elif label == 'neither':
# 				ex[0]='neutral'
# 				neutrals.append(ex)
	
# 	train,test = shuffle_split(positives, negatives, neutrals)

# 	with open(DATA_OUT+"RWA-ISIS_train.txt","w") as fod:
# 			for ex in train:
# 				fod.write('\t'.join(ex)+"\n")

# 	with open(DATA_OUT+"RWA-ISIS_test.txt","w") as fod:
# 		for ex in test:
# 			fod.write('\t'.join(ex)+"\n")

# 	return train + test

def get_parser():
    parser = argparse.ArgumentParser(description="Linear Classifier")
    
    #Basic Input
    parser.add_argument('-bin', action="store_true", default=False,
                        help='if True, only positive and negatives will be used')
    return parser

if __name__ == "__main__":

	parser = get_parser()
	args = parser.parse_args()    

	if args.bin:
		print "Binary labels"
	instances = []
	instances += get_omd(args.bin)
	instances += get_hcr(args.bin)
	instances += get_semeval(args.bin)
	msgs = [inst[1] for inst in instances]
	wrd2idx = word_2_idx(msgs)
	with codecs.open("DATA/txt/vocab.txt","w","utf-8") as fid:
		for w in wrd2idx.keys():
			fid.write(w+"\n")


# save_embeddings_txt("DATA/embeddings/embs_400.txt", "DATA/embeddings/filtered_embs.txt", wrd2idx,init_ooe=True)
# set_trace()
