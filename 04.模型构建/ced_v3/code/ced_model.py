# encoding: utf-8

import os
import gc
import random
import numpy as np
import tensorflow as tf
from keras.utils import np_utils


# 配置路径
data_save_path = '/home/qhli/CED/ced_cnn/data/'


# 参数定义
tf.app.flags.DEFINE_float('alpha', 0.875, 'threshold')		# 概率阈值 α
FLAGS = tf.app.flags.FLAGS

α = FLAGS.alpha
hidden_size = 128			# 隐层单元数
batch_size = 256				
vocabulary_size = 1000
time_steps = 1000
epochs = 30
lambda0 = 0.1				# 目标函数超参数 
lambda1 = 0.2


# 自定义目标函数
def ced_loss(y_true, y_pred, alpha=α, λ0=0.1, λ1=0.2):
	y = tf.cast(tf.argmax(y_true, 2), 'float32')
	# 计算达到 α 的时间步
	temp = tf.cumsum(tf.reduce_sum(tf.cast(tf.greater_equal(y_pred, alpha), 'int32'), axis=2), axis=1)
	temp = tf.cast(tf.cast(temp, 'bool'), 'float32')
	# shape(batch_size, time_stpes)
	nf = tf.reduce_sum((1-temp), axis=1)
	# shape(batch_size)

	# 计算 β 
	beta = tf.cast((nf / time_steps), 'float32')
	k = 1 / ((1-beta+1e-9) * time_steps)
	# shape(batch_size, )

	O_time = tf.negative(tf.log(beta))
	# shape(batch_size, )

	# 计算样例的每部分误差
	O_i = tf.log(tf.clip_by_value(tf.reduce_sum((y_true*y_pred), axis=2), 1e-10, 1.0))
	# shape(batch_size, time_steps)

	diff = (y*tf.maximum(0.0, tf.log(alpha)-O_i)) + (1-y)*tf.maximum(0.0, O_i-tf.log(1-alpha))
	O_diff = tf.negative(k) * tf.reduce_sum((diff*temp), axis=1)
	O_pred = k * tf.reduce_sum((O_i*temp), axis=1)
	O_ced = O_pred 

	return -tf.reduce_mean(O_ced)


# 自定义评价函数
def ced_metrics(y_true, y_pred, alpha=α):
	# 提取样例的真实类别
	y = tf.argmax(y_true, 2)
	y_ = tf.argmax(y_pred, 2)

	# 计算达到 α 的时间步
	temp = tf.cumsum(tf.reduce_sum(tf.cast(tf.greater_equal(y_pred, alpha), 'int32'), axis=2), axis=1)
	temp = tf.cast(tf.cast(temp, 'bool'), 'int32')
	nf = tf.reduce_sum((1-temp), axis=1) - 1
	# shape(batch_size, )

	a = 0.0
	# 计算是否正确预测
	for n in range(batch_size):
		a = tf.cond(tf.equal(y[n, nf[n]], y_[n, nf[n]]), lambda: a+1, lambda: a)

	# 计算正确率
	accuracy = a / batch_size
	return accuracy


# 获取一个 batch_size 大小的数据
def get_batches(num, batch_size):
    for i in range(0, num, batch_size):
        begin_i = i
        end_i = i+batch_size if (i+batch_size) < num else num
        # yield 来连续获取大小为 batch_size 的数据
        yield begin_i, end_i


def softmax(inputs):
	_sum = tf.reduce_sum(inputs, reduction_indices=2, keepdims=True) + 1e-9  # batch_size * 1 * 1
	return inputs / _sum  # batch_size * 1 * max_sen_len


def main(_):

	X = tf.placeholder(tf.float32, [None, time_steps, vocabulary_size], name='input_fi')
	Y = tf.placeholder(tf.float32, [None, time_steps, 2], name='input_label')
	S = tf.placeholder(tf.float32, [None], name='sequence_length')

	gru = tf.contrib.rnn.GRUCell(hidden_size)

	outputs, _ = tf.nn.dynamic_rnn(gru, X, S, dtype=tf.float32)

	y_pred = softmax(tf.contrib.layers.fully_connected(outputs, 2))

	loss = ced_loss(Y, y_pred)

	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	#accuracy = ced_metrics(Y, y_pred)


	# 读取预处理好的数据
	print('Reading Data------------------------------')
	fi_train = np.asarray(np.load(data_save_path + 'input_fi_train.npy'))
	label_train = np.asarray(np.load(data_save_path + 'input_label_train.npy'))
	seq_train = np.asarray(np.load(data_save_path + 'sequence_length_train.npy'))
	
	label_train = [[i]*time_steps for i in label_train]
	label_train = np_utils.to_categorical(label_train, num_classes=2)

	print('fi_train: ', fi_train.shape)
	print('label_train: ', label_train.shape)
	print('seq_train: ', seq_train.shape)

	with tf.Session() as sess:
    	# 初始化变量
		tf.global_variables_initializer().run()
		
		for e in range(1):
			# 模型训练
			print('Training------------------------------')
			for begin, end in get_batches(len(seq_train), batch_size):
				feed_dict = {X: fi_train[begin: end], Y: label_train[begin: end], S: seq_train[begin: end]}
				train_loss, pred, _ = sess.run([loss, y_pred, optimizer], feed_dict=feed_dict)
				print(train_loss)

			print('Epochs:{}/{}'.format(e, epochs), 'Train loss: {:.8f}'.format(train_loss))
			

if __name__ == "__main__":
	tf.app.run()
