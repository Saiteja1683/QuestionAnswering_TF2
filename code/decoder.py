from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest




def BiLSTM( inputs, masks, scope_name, dropout, hidden_states):
	with tf.compat.v1.variable_scope(scope_name):
		lstm_fw_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_states),output_keep_prob = dropout)
		lstm_bw_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_states),output_keep_prob = dropout)
		seq_len = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)

		outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs = inputs, sequence_length = seq_len, dtype = tf.float64)
		hidden_outputs = tf.concat(outputs, 2)
	return hidden_outputs, seq_len


def output_layer( G, M, masks, dropout, scope_name, hidden_states):
        # M (?, m, 2h)
        # the softmax part is implemented together with loss function
	with tf.compat.v1.variable_scope(scope_name):

		w_1 = tf.compat.v1.get_variable('w_start', shape=(10 * hidden_states, 1), dtype = tf.float64,
		initializer=tf.keras.initializers.GlorotNormal())
		batch_size = tf.shape(M)[0]
		w_1_tiled = tf.tile(tf.expand_dims(w_1, 0), [batch_size, 1, 1])
		print(w_1_tiled,'tilll')
		print(G,'ggggg')
		print(M,'mmm')
		# w_2 = tf.compat.v1.get_variable('w_end', shape=(10 * hidden_states, 1), dtype = tf.float64,
		# initializer=tf.keras.initializers.GlorotNormal())


		

		M2, _ = BiLSTM(M, masks, scope_name, dropout, hidden_states)
		print(M2)
		
		temp1 = tf.concat([G, M], 2)  # (?, m, 10h)
		print(temp1,'10d')
		#temp2 = tf.concat([G, M2], 2)  # (?, m, 10h)
		#temp_1_o = tf.nn.dropout(temp1, dropout)
		temp_2_o = tf.nn.dropout(temp1, dropout)
		print(temp_2_o,'temp2')

		
		#w_2_tiled = tf.tile(tf.expand_dims(w_2, 0), [batch_size, 1, 1])
	
		pred_s = tf.squeeze(tf.einsum('aij,ajk->aik',temp_2_o, w_1_tiled)) # (?, m, 10h) * (?, 10h, 1) -> (?, m, 1)
		print(pred_s,'predddddd')
		#print(tf.compat.v1.layers.Dense(pred_s))
		#logits = tf.transpose(tf.keras.layers.Dense(pred_s),perm=[1, 0, 2], use_bias=False)
		
		#pred_e = tf.squeeze(tf.einsum('aij,ajk->aik',temp_2_o, w_2_tiled)) # (?, m, 10h) * (?, 10h, 1) -> (?, m, 1)
		return pred_s


def decode(G, context_mask_placeholder, dropout, hidden_states):

	M, _c = BiLSTM( G, context_mask_placeholder,  'model_layer',dropout, hidden_states)
	pred_s  = output_layer( G, M, context_mask_placeholder, dropout, 'output_layer', hidden_states)
	return pred_s 


