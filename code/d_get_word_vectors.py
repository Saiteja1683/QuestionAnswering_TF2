import numpy as np
import nltk
from b_context_words_to_num import *


#answer = [line.rstrip('\n') for line in open('./data/data/answers.txt')]

def Glove(file_cont,file_que,cont_mxm_lgth,que_mxm_lgth,hidden_states):

	with open("glove.6B.50d.txt",'r',encoding="utf-8") as f1:
		kk = f1.readlines()
		
	with open(file_cont,'r',encoding="utf-8") as f1:
		contexts = f1.readlines()
		
	with open(file_que,'r',encoding="utf-8") as f1:
		questions = f1.readlines()

	dup=[]
	que=[]
	for i,line in enumerate(questions):
		context_line = nltk.word_tokenize(line.lower().strip())
		dup.extend(context_line)
		que.append(context_line)  

	cont=[]
	for i,line in enumerate(contexts):
		pre_processed_line = normalize_corpus([line], text_lemmatization=False, stopword_removal=True, text_lower_case=True)
		processed_tokens = word_2_numbers(pre_processed_line)
		final_context = []
		for toks in processed_tokens:
			try:
				final_context.append(text2int(toks))
			except:
				final_context.append(toks)
		
		dup.extend(final_context)
		cont.append(final_context)

	un = list(set(dup))
	print(len(un))
	one = np.ones([50])
	print(one)
	glo = {}
	for line in kk:
		line = line.strip()
		line = line.split()
		a = line[0]
		b = np.asarray(line[1:])
		glo[a] = b

	words_in_glove = glo.keys()
	unique={}

	for word in un:
		if word in words_in_glove:
			unique[word] = glo[word]
		else:
			unique[word] = one


	a = np.zeros([len(cont),cont_mxm_lgth,hidden_states], dtype = float)
	b = np.zeros([len(que),que_mxm_lgth,hidden_states], dtype = float)

	for i in range(len(cont)):
		for j in range(len(cont[i])):
			kk = (unique[cont[i][j]])
			#print(kk.shape)
			a[i][j]=kk	

	for i in range(len(que)):
		for j in range(len(que[i])):
			kk = (unique[que[i][j]])
			b[i][j]=kk	
	 			
	return a,cont,b,que


def ans(answer_file,context):
	print(answer_file)
	#print(len(context))
	with open(answer_file,"r") as f1:
		answer = f1.readlines()
	#print(answer)
	p = 0
	temp = []
	for num in range(len(context)):
		ans_l = answer[num].strip('\n').lower()
		con_l = context[num]
		#print(num)
		contains_digit = any(map(str.isdigit, ans_l))
		check_lis_1 = []
		
		check_lis_1 = (ans_l.split('.'))
		check_lis_1.extend(ans_l)
		check_lis_1.extend(ans_l.split(' '))
		#print(check_lis_1)
		try:
			val = next(element for element in check_lis_1 if element in con_l)
		except:
			for k in check_lis_1:
				if str(k) in con_l:
					#print('cccc')
					val = k
		idx = con_l.index(val)
		temp.append(idx)
		
	
	return temp

# cont_mxm_lgth = 4000
# que_mxm_lgth = 50
# hidden_states = 50

# a,cont,b,que = Glove('./data/data/contexts.txt','./data/data/questions.txt',cont_mxm_lgth,que_mxm_lgth,hidden_states)
# print(len(a))
# print(len(a[0]))
# print(len(a[0][0]))

# print(len(b))
# print(len(b[0]))
# print(len(b[0][0]))
