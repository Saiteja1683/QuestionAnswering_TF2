from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging

import numpy as np
import tensorflow as tf

#from tensorflow.contrib import rnn
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



def BiLSTM(inputs, masks, scope_name, dropout, hidden_states):
	with tf.compat.v1.variable_scope(scope_name):
		lstm_fw_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_states),output_keep_prob = dropout)
		lstm_bw_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_states),output_keep_prob = dropout)
		seq_len = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
		outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs = inputs, sequence_length = seq_len, dtype = tf.float64)
		hidden_outputs = tf.transpose(tf.concat(outputs, 2), perm=[0, 2, 1])
	return hidden_outputs


def bi_attention( y_q, y_c, hidden_states, max_cont_lgth):
        # y_q: (?, 2h, n)
        # y_c: (?, 2h, m)
        # need to compute S first
        # S: (?, m, n)
	with tf.compat.v1.variable_scope('bi_attention') as scope:
		S = bilinear_similarity( y_q, y_c, hidden_states)
		H = Q2C_attention( y_q, y_c, S, max_cont_lgth)  # H = (?, 2h, m)
		U = C2Q_attention( y_q, y_c, S)  # U = (?, 2h, m)
            # need to compute G
		G = tf.concat([y_c, H, y_c * U, y_c * H], 1)  # G = (?, 8h, m)
		G = tf.transpose(G, perm=[0, 2, 1]) # (?, m, 8h)
	return G


def bilinear_similarity(y_q, y_c, hidden_states):
        # y_q: (?, 2h, n)
        # y_c: (?, 2h, m)
        # S : (?, n, m)

	with tf.compat.v1.variable_scope('similarity') as scope:
		batch_size = tf.shape(y_c)[0]
		w_alpha = tf.compat.v1.get_variable('w_alpha', shape=(2 * hidden_states, 2 * hidden_states), dtype = tf.float64, initializer=tf.keras.initializers.GlorotNormal())

		w_alpha_tiled = tf.tile(tf.expand_dims(w_alpha, 0), [batch_size, 1, 1])
		y_q_T = tf.transpose(y_q, perm=[0, 2, 1]) # U_T: (?, n, 2h)
		bi_S_temp = tf.einsum('aij,ajk->aik', y_q_T, w_alpha_tiled) # (?, n, 2h) * (2h, 2h) = (?, n, 2h)
		S = tf.einsum('aij,ajk->aik', bi_S_temp, y_c)  # (?, n, 2h) * (?, 2h, m) = (?, n, m)
	return S


def C2Q_attention( y_q, y_c, S):
        # y_q: (?, 2h, n)
        # y_c: (?, 2h, m)
        # S: (?, n, m)
	a = tf.nn.softmax(S, axis=1)   # (?, n, m)
	U = tf.matmul(y_q, a)    # (?, 2h, n) * (?, n, m) = (?, 2h, m)
	return U

def Q2C_attention( y_q, y_c, S, max_cont_lgth):
        # y_q: (?, 2h, n)
        # y_c: (?, 2h, m)
        # S: (?, n, m)
	b = tf.nn.softmax(tf.reduce_max(S, axis=1, keepdims=True)) # b = (?, 1, m)
	h = tf.einsum('aij,akj->aik', y_c, b) # (?, 2h, m) * (?, 1, m) => (?, 2h, 1)
	H = tf.tile(h, [1, 1, max_cont_lgth])
	return H


def encode( context, context_mask_placeholder, question, question_mask_placeholder, dropout, hidden_states, max_cont_lgth):

	yc = BiLSTM( context , context_mask_placeholder, 'context_BiLSTM', dropout, hidden_states) # (?, 2h, m)

	yq = BiLSTM( question, question_mask_placeholder, 'question_BiLSTM', dropout, hidden_states) # (?, 2h, n)


	return bi_attention( yq, yc, hidden_states, max_cont_lgth)
