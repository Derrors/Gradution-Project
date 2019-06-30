# encoding: utf-8

import os
import re
import gc
import math
import json
import jieba
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, GRU
from keras.layers import Activation
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer


# 配置路径
weibo_json_path = '../data/Weibo/json/'
weibo_label_path = '../data/Weibo/label.txt'
data_save_path = '/home/qhli/1080/CED/ced_cnn/data/Weibo/'


# 参数定义
tf.app.flags.DEFINE_float('alpha', 0.975, 'threshold')		# 概率阈值 α
FLAGS = tf.app.flags.FLAGS

α = FLAGS.alpha
N = 10						# 转发聚合度
hidden_size = 200			# 隐层单元数
batch_size = 10				
vocabulary_size = 1000
time_steps = 1000
lambda0 = 0.1				# 目标函数超参数 
lambda1 = 0.2


# 过滤文本信息中的非中文字符
def str_filter(string):
	chars = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
	string = ''.join(string.split())
	return re.sub(chars, '', string)


# 获取事件列表，格式为：eid, label, posts_num
def get_eventlist():
	event_list = []
	fl = open(weibo_label_path, 'r')
	lines = fl.readlines()
	for line in lines:
		event_str = line.split()
		eid = (event_str[0].split(':'))[1]
		label = (event_str[1].split(':'))[1]
		posts_num = len(event_str[2:])
		event = {'eid': eid, 'label': label, 'posts_num': posts_num}
		event_list.append(event)
	fl.close()
	return event_list


# 将文本分词并转换为词向量表示
def get_word_vector():
	input_fi = []
	input_label = []
	reposts = []
	count = []
	stop_words = [
		'一下', '一个', '一些', '一份', '一位', '一切', '一句', '一只', 
		'一台', '一场', '一天', '一定', '一年', '一张', '一条', '一样', 
		'一次', '一点', '一直', '一看', '一种', '一群', '一起', '一路', 
		'三个'
		]
	event_list = get_eventlist() 
	
	for event in event_list:
		label = []
		for i in range(time_steps):
			label.append([event['label']])
		input_label.append(label)
		count.append(math.ceil(event['posts_num'] / 10))
		event_json_path = os.path.join(weibo_json_path, event['eid'])

		fj = open(event_json_path, 'r')
		event_json = json.load(fj)
		for i in range(0, event['posts_num'], N):
			sentence = ''
			w = []
			for q in range(N):
				if i+q < event['posts_num']:	
					weibo = event_json[i+q]
					text = weibo['text']

					text = str_filter(text)                         # 文本过滤
					words = jieba.cut(text, cut_all=False)          # 分词
					for word in words:
						w.append(word)
			w = set(w)
			for item in w:
				sentence = sentence + ' ' + item
			reposts.append(sentence)
			del sentence
		fj.close()

	vectorizer = TfidfVectorizer(max_features=vocabulary_size, stop_words=stop_words)
	re = vectorizer.fit_transform(reposts)
	re = re.todense()

	index = 0
	for n in count:
		fi = []
		if n >= time_steps:
			fi = (re[index: index+time_steps]).tolist()
		else:
			fi = (re[index: index+n]).tolist()
			for q in range(time_steps-n):
				fi.append(np.zeros((vocabulary_size, ), 'float32'))
		input_fi.append(fi)
		index += n
	del reposts, count, re
	gc.collect()

	return input_fi, input_label


def get_word_vector_v2():
	input_fi = []
	input_label = []
	reposts = []
	count = []
	stop_words = [
		'一下', '一个', '一些', '一份', '一位', '一切', '一句', '一只', 
		'一台', '一场', '一天', '一定', '一年', '一张', '一条', '一样', 
		'一次', '一点', '一直', '一看', '一种', '一群', '一起', '一路', 
		'三个'
		]
	event_list = get_eventlist()

	for event in event_list:
		input_label.append(event['label'])
		count.append(math.ceil(event['posts_num'] / 10))
		event_json_path = os.path.join(weibo_json_path, event['eid'])

		fj = open(event_json_path, 'r')
		event_json = json.load(fj)
		for i in range(0, event['posts_num'], N):
			sentence = ''
			w = []
			for q in range(N):
				if i+q < event['posts_num']:	
					weibo = event_json[i+q]
					text = weibo['text']

					text = str_filter(text)                         # 文本过滤
					words = jieba.cut(text, cut_all=False)          # 分词
					for word in words:
						w.append(word)
			w = set(w)
			for item in w:
				sentence = sentence + ' ' + item
			reposts.append(sentence)
			del sentence
		fj.close()

	vectorizer = TfidfVectorizer(max_features=vocabulary_size, stop_words=stop_words)
	re = vectorizer.fit_transform(reposts)
	re = re.todense()

	index = 0
	sequence_length = []
	for n in count:
		fi = []
		if n >= time_steps:
			fi = (re[index: index+time_steps]).tolist()
			sequence_length.append(time_steps)	
		else:
			fi = (re[index: index+n]).tolist()
			for q in range(time_steps-n):
				fi.append(np.zeros((vocabulary_size, ), 'float32'))
				sequence_length.append(n)
		input_fi.append(fi)
		index += n
	
	del reposts, count, re
	gc.collect()
	
	embeddings = []
	embeddings_index = {}

	id = 1
	embeddings.append(np.zeros((vocabulary_size, ), 'float32'))
	for i in range(len(input_fi)):
		for q in range(time_steps):
			sent = input_fi[i][q]
			if sent not in embeddings:
				embeddings.append(sent)
				embeddings_index[id] = sent
				id += 1

	print(id)

	for i in range(len(input_fi)):
		for q in range(time_steps):
			sent = input_fi[i][q]
			input_fi[i][q] = embeddings.index(sent)
	
	np.save(data_save_path + 'input_fi.npy', input_fi)
	np.save(data_save_path + 'input_label.npy', input_label)
	np.save(data_save_path + 'sequence_length.npy', sequence_length)
	np.save(data_save_path + 'embeddings_index.npy', embeddings_index)

	del input_fi, input_label, sequence_length, embeddings, embeddings_index
	gc.collect()
	
	print('Data preprocessing completed.')

	return input_fi, input_label


# 自定义目标函数
def ced_loss(y_true, y_pred, alpha=α, λ0=0.1, λ1=0.2):
	y = tf.reduce_sum(tf.cast(tf.equal(y_true, [0, 1]), 'float32'), axis=2)
	y = tf.cast(tf.cast(y, 'bool'), 'float32')
	# shape(batch_size, time_steps, )

	# 计算达到 α 的时间步
	temp = tf.cumsum(tf.reduce_sum(tf.cast(tf.greater_equal(y_pred, alpha), 'int32'), axis=2), axis=1)
	temp = tf.cast(tf.cast(temp, 'bool'), 'float32')
	# shape(batch_size, time_stpes)
	nf = tf.reduce_sum((1-temp), axis=1)
	# shape(batch_size)

	# 计算 β 
	beta = tf.cast((nf / time_steps), 'float32')
	k = 1 / ((1-beta) * time_steps)
	# shape(batch_size, )

	O_time = tf.negative(tf.log(beta))
	# shape(batch_size, )

	# 计算样例的每部分误差
	O_i = tf.log(tf.reduce_sum((y_true * y_pred), axis=2))
	# shape(batch_size, time_steps)

	diff = (y*tf.maximum(0.0, tf.log(alpha)-O_i)) + (1-y)*tf.maximum(0.0, O_i-tf.log(1-alpha))
	O_diff = tf.negative(k) * tf.reduce_sum((diff*temp), axis=1)
	O_pred = k * tf.reduce_sum((O_i*temp), axis=1)
	O_ced = O_pred + λ0 * O_diff + λ1 * O_time

	return tf.reduce_mean(O_ced)
	

# 自定义评价函数
def ced_metrics(y_true, y_pred, alpha=α):
	# 提取样例的真实类别
	y = tf.reduce_sum(tf.cast(tf.equal(y_true, [0, 1]), 'int32'), axis=2)
	y = tf.cast(tf.cast(y, 'bool'), 'int32')
	# shape(batch_size, time_steps, )
	y = y[:, 0]
	# shape(batch_size, )

	# 计算达到 α 的时间步
	temp = tf.cumsum(tf.reduce_sum(tf.cast(tf.greater_equal(y_pred, alpha), 'int32'), axis=2), axis=1)
	temp = tf.cast(tf.cast(temp, 'bool'), 'int32')
	nf = tf.reduce_sum((1-temp), axis=1) - 1
	# shape(batch_size, )

	a = 0
	# 计算是否正确预测
	for n in range(batch_size):
		cond1 = tf.cast(tf.equal(y[n], 0), 'int32') + tf.cast(tf.greater_equal(y_pred[n, nf[n], 0], alpha), 'int32')
		cond2 = tf.cast(tf.equal(y[n], 1), 'int32') + tf.cast(tf.greater_equal(y_pred[n, nf[n], 1], alpha), 'int32')
		a = tf.cond(tf.equal(cond1, 2), lambda: a+1, lambda: a)
		a = tf.cond(tf.equal(cond2, 2), lambda: a+1, lambda: a)

	# 计算正确率
	accuracy = a / batch_size
	return accuracy


# 获取一个 batch_size 大小的数据
def get_batches_random(x, y, batch_size):
    length = len(y)
    loop_count = length // batch_size
    while (True):
        i = np.random.randint(0, loop_count)
        yield x[i * batch_size: (i+1) * batch_size], y[i * batch_size: (i+1) * batch_size]


def main(_):
	'''
	# get_word_vector_v2()
	
	input_fi, input_label = get_word_vector()

	# 划分训练集与测试集
	fi_train = input_fi[: int(0.75*len(input_fi))]
	np.save(data_save_path + 'fi_train.npy', fi_train)
	del fi_train
	gc.collect()

	fi_test = input_fi[int(0.75*len(input_fi)): ]
	np.save(data_save_path + 'fi_test.npy', fi_test)
	del fi_test,input_fi
	gc.collect()

	label_train = input_label[: int(0.75*len(input_label))]
	label_test = input_label[int(0.75*len(input_label)): ]
	np.save(data_save_path + 'label_train.npy', label_train)
	np.save(data_save_path + 'label_test.npy', label_test)
	del label_train, label_test, input_label
	gc.collect()
	'''

	fi_train = np.array(np.load(data_save_path + 'fi_train.npy'))
	label_train = np.array(np.load(data_save_path + 'label_train.npy'))
	label_train = np_utils.to_categorical(label_train, num_classes=2)
	
	print('Data preprocessing completed----------')

	# GRU model
	model = Sequential()

	model.add(GRU(hidden_size, input_shape=(time_steps, vocabulary_size), return_sequences=True))
	
	model.add(Dense(2, activation='softmax'))

	model.compile(optimizer=Adam(), loss=ced_loss, metrics=['binary_accuracy'])

	# 模型训练
	print('Training------------------------------')

	model.fit_generator(generator=get_batches_random(fi_train, label_train, batch_size), 
						steps_per_epoch=1,
						epochs=10)
	
	# model.fit(fi_train, label_train, batch_size=batch_size, epochs=30)
	'''
	del fi_train, label_train
	gc.collect()
	
	fi_test = np.array(np.load(data_save_path + 'fi_test.npy'))
	label_test = np.array(np.load(data_save_path + 'label_test.npy'))
	label_test = np_utils.to_categorical(label_test, num_classes=2)

	# 模型测试
	print('Testing------------------------------')

	loss_and_accuracy = model.evaluate(fi_test, label_test, batch_size=batch_size, verbose=0)
	print('test loss: ', loss_and_accuracy[0])
	print('test accuracy: ', loss_and_accuracy[1])
	'''	

if __name__ == "__main__":
	tf.app.run()
