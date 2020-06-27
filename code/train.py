import os
import numpy as np
import pandas as pd
import tensorflow as tf
from d_get_word_vectors import Glove,ans
from encoder import encode
from decoder import decode
from collections import Counter
import re
tf.compat.v1.disable_eager_execution()

file1 = open('../output/result1.txt','w')
file2 = open('../output/result2.txt','w')


total_size = 252
batch_size = 4
hidden_states = 50
max_para_lgth = 4000
que_mxm_lgth = 50
drop_out = 0.5
learning_rate = 0.02
train_size = 200
test_size = 52
max_gradient_norm = 5.0
   
global cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder, dropout_placeholder, model, updates,loss,global_step,param_norm,gradient_norm

############################

def pad_sequence(data, max_length):
	mask = []
	ret = []
		# Use this zero vector when padding sequences.
	zero_label = 0

	for sentence in data:
		pad_num = max_length - len(sentence)
		if pad_num > 0:
			#ret.append(sentence[:] + [zero_label] * pad_num)
			mask.append([True] * len(sentence) + [False] * pad_num)
		else:
			#ret.append(sentence[:max_length])
			mask.append([True] * max_length)
		#print(len(sentence),max_length,sum(mask[sentence]))

	return mask


################## test functions #################3
def final_spans(span_ss):

	start_span = np.argmax(span_ss, 1)

	return start_span

def test(session, test_data,pred_ans,cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder):

	input_feed = create_feed_dict(test_data[0:4],cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder)
	output_feed = [pred_ans]
	p  = session.run(output_feed, input_feed)
	#print(_s)
	#print(_de)
	return p

def test_batches(s, test_data,pred_ans,cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder):
	num_batches = int(len(test_data[0]) / batch_size)
	
	print("-------------------------------------------------------------------------")
	overall_f1 = 0
	overall_em = 0
	for i in range(num_batches):
		batch_f1 = 0
		batch_em = 0
		ct = i*batch_size
		batch = []
		contt = test_data[0][ct:ct+batch_size]
		que = test_data[1][ct:ct+batch_size]
		contt_mask = test_data[3][ct:ct+batch_size]
		quee_mask = test_data[4][ct:ct+batch_size]	

		batch.append(contt)
		batch.append(que)
		batch.append(contt_mask)
		batch.append(quee_mask)
		
		labels = test_data[5][ct:ct+batch_size]
		trues = test_data[2][ct:ct+batch_size]
		
		xx = test(s, batch,pred_ans,cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder)

		a = final_spans(xx)

		
		k = 0
		
		for (a_s, context, a_true) in zip(a, labels, trues):
		
			predicted_answer = formulate(context, a_s)
			true_answer = formulate(context, a_true)
			file1.write(predicted_answer+'\n')
			file2.write(true_answer+'\n')
			f1 = f1_score(predicted_answer, true_answer)
			batch_f1 += f1
			if exact_match_score(predicted_answer, true_answer):
				batch_em += 1
		
		batch_f1 = batch_f1/100
		batch_em = batch_em/100
		print("batch no: ",i+1,"f1 score: ",batch_f1)
		print("batch no: ",i+1,"em: ",batch_em)
		overall_f1 += batch_f1
		overall_em += batch_em


	overall_f1 = overall_f1 / num_batches
	overall_em = overall_em / num_batches
	print("-------------------------------------------------------------------------")
	return overall_f1,overall_em

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    # def remove_punc(text):
    #     exclude = set(string.punctuation)
    #     return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(s)))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def formulate(data, s):
	#ls = word_tokenize(data)
	ans= ''
#	i=int(s)

	try:
		ans = ans+' '+data[s]
	except:
		print("total_length: ",len(data))		

	return ans

def create_feed_dict(data_batch,cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder):
    
    if len(data_batch) == 4:
        feed_dict = {
            cont_plce_hlder : data_batch[0],
            que_plce_hlder : data_batch[1],
            context_mask_placeholder : data_batch[2],
            question_mask_placeholder : data_batch[3]
            }
    else:
        feed_dict = {
		cont_plce_hlder : data_batch[0],
		que_plce_hlder : data_batch[1],
        ans_plce_hlder : data_batch[2],
		context_mask_placeholder : data_batch[3],
		question_mask_placeholder : data_batch[4]
		
	}
    return feed_dict


def network(cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder, dropout_placeholder):

    G= encode(cont_plce_hlder, context_mask_placeholder ,que_plce_hlder, question_mask_placeholder, drop_out,hidden_states, max_para_lgth)
    pred_ans = decode(G, context_mask_placeholder, drop_out,hidden_states)

    print(pred_ans)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ans_plce_hlder,logits=pred_ans))

    params = tf.compat.v1.trainable_variables()
    gradients = tf.gradients(loss, params)
    gradient_norm = tf.linalg.global_norm(gradients)
    clipped_gradients,_ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    param_norm = tf.linalg.global_norm(params)

    global_step = tf.Variable(0,name="global_step", trainable=False)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate)
    updates = opt.apply_gradients(zip(clipped_gradients, params), global_step = global_step)

    return updates,loss,global_step,param_norm,gradient_norm,pred_ans

def optimize(session,model, batch,cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder):
  
	input_feed = create_feed_dict(batch,cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder)
	lis_up = session.run(model, input_feed)

	return lis_up[1]


def getData(context_file,questions_file,answers_file,flag):
	cont_emb,cont,que_emb,que = Glove(context_file,questions_file,max_para_lgth,que_mxm_lgth,hidden_states)
	ans_list = ans(answers_file,cont)

	cont_mask = pad_sequence(cont ,max_para_lgth )	
	que_mask = pad_sequence(que ,que_mxm_lgth )
	train_data = []
	train_data.append(cont_emb[0: train_size])
	train_data.append(que_emb[0: train_size])
	train_data.append(ans_list[0: train_size])
	train_data.append(cont_mask[0: train_size])
	train_data.append(que_mask[0: train_size])

	if flag:
		train_data.append(cont[0: train_size])

	
	return train_data


def batch_wise_train(sess, model, train_data, e, cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder):
	num_batches = int(len(train_data[0]) / batch_size)

	for i in range(num_batches):
		ct = i*batch_size
		batch = []
		contt = train_data[0][ct:ct+batch_size]
		que = train_data[1][ct:ct+batch_size]
		span_s = train_data[2][ct:ct+batch_size]
		contt_maskk = train_data[3][ct:ct+batch_size]
		quee_maskk = train_data[4][ct:ct+batch_size]	

		batch.append(contt)
		batch.append(que)
		batch.append(span_s)
		batch.append(contt_maskk)
		batch.append(quee_maskk)
		
		loss = optimize(sess, model,batch,cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder)
		print(i+1, loss, e+1)
	return loss



train_context_file = '../data/train_txts/context.txt'
train_questions_file = '../data/train_txts/questions.txt'
train_answers_file = '../data/train_txts/answers.txt'

test_context_file  = '../data/test_txts/context.txt'
test_questions_file = '../data/test_txts/questions.txt'
test_answers_file = '../data/test_txts/answers.txt'

def main():
    
    flag = False
    train_data = getData(train_context_file,train_questions_file,train_answers_file,flag)

    cont_plce_hlder = tf.compat.v1.placeholder(tf.float64, [batch_size, max_para_lgth, hidden_states])
    que_plce_hlder = tf.compat.v1.placeholder(tf.float64, [batch_size, que_mxm_lgth, hidden_states])
    context_mask_placeholder = tf.compat.v1.placeholder(tf.bool, shape=(batch_size,max_para_lgth))
    question_mask_placeholder = tf.compat.v1.placeholder(tf.bool, shape=(batch_size,que_mxm_lgth))
    ans_plce_hlder = tf.compat.v1.placeholder(tf.int32, [None])
    dropout_placeholder = tf.compat.v1.placeholder(tf.float64)

    model = network(cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder, dropout_placeholder)
    
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as s:
        s.run(init)
        for epoch in range(100):
            final_loss = batch_wise_train(s,model, train_data, epoch, cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder)

            if epoch%20==0:
                test_data = getData(test_context_file,test_questions_file,test_answers_file,True)
                f1,em = test_batches(s,test_data,model[-1],cont_plce_hlder, que_plce_hlder, context_mask_placeholder, question_mask_placeholder, ans_plce_hlder)

                print("#####################################################################")
                print("\nfinal f1 score: ",f1)
                print("final em: ",em,"\n")

                print("#####################################################################")

                

main()
