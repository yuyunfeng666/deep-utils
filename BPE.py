import re, collections
import sys

def get_stats(vocab):
	pairs = collections.defaultdict(int)
	for word,freq in vocab.items():
		symbols = word.split()
		for i in range(len(symbols)-1):
			pairs[symbols[i],symbols[i+1]]+=freq
	return pairs
def merge_vocab(pair,v_in):
	v_out={}
	bigram=re.escape(' '.join(pair))
	p = re.compile(r'(?<!\S)'+bigram+r'(?!\S)')
	for word in v_in:
		w_out = p.sub(''.join(pair),word)
		v_out[w_out]=v_in[word]
	return v_out
if __name__=='__main__':
	res = []
	num_merges = 15
	file1 = open('test','rb')
	vocab = {}
	ind = {}
	num=0
	ind_inv={}
	line_num=0
	for line in file1.readlines():
		line = line.rstrip().split()
		for term in line:
			term = term.decode('utf8')
			if len(term)==0:
				continue
			if term not in ind:
				ind[term]=[]
			ind[term].append(num)
			ind_inv[num]=term
			num+=1
			s = ''
			for i in term[:-1]:
				s+=i
				s+=' '
			s+=term[-1]
			if s not in vocab:
				vocab[s]=0
			vocab[s]+=1
		ind_inv[num]='<\s>'
		num+=1
		line_num+=1
		if line_num%1000==0:
			print(line_num)
	print(len(vocab))
	print(vocab)
	for i in range(num_merges):
		print(i)
		pairs =get_stats(vocab)
		best = max(pairs,key=pairs.get)
		vocab = merge_vocab(best,vocab)
	print(vocab)
	out=[0]*num
	print(num,len(vocab),len(ind))
	fileout = open('res','wb')
	for (k,v) in vocab.items():
		s =''.join( k.split(' '))
		for i in ind[s]:
			out[i]=k.split(' ')
	for i in range(num):
		if out[i] not in range(0,9):
			for j in out[i]:
				if j not in res:
					res.append(j)
print(res)
