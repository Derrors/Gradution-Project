# encoding:utf-8

import os
import re
import json
import jieba
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, GRU, Dropout
from keras.optimizers import Adam, Adagrad
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# 配置路径信息
weibo_json_path = '../../Weibo/json/'
weibo_label_path = '../../Weibo/label.txt'
weibo_embedding_path = '../data/embedding/weibo.json'

# 配置参数信息
tf.app.flags.DEFINE_integer('Hours', 1000, "deadline")
FLAGS = tf.app.flags.FLAGS

time_steps = 50 					# 时间步数
deadline = FLAGS.Hours * 3600			# Deadline
embedding_size = 300
batch_size = 128
output_size = 2                         # 输出维度
cell_num = 200                          # 隐层单元数
learning_rate = 0.001                   # 学习率


# 过滤文本信息中的非中文字符
def str_filter(string):
	chars = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
	string = ''.join(string.split())
	return re.sub(chars, '', string)


# 获取事件列表，格式为：eid, label, posts_num
def get_eventlist():
	event_list = []
	with open(weibo_label_path, 'r') as fl:
		lines = fl.readlines()
		for line in lines:
			event_str = line.split()
			eid = (event_str[0].split(':'))[1]
			label = (event_str[1].split(':'))[1]
			posts_num = len(event_str[2:])
			event = {'eid': eid, 'label': label, 'posts_num': posts_num}
			event_list.append(event)
	return event_list


# 获取最大时间跨度的连续时间间隔
def get_continuous_intervals(intervals):
	max_int = []
	temp = [intervals[0]]
	for q in range(1, len(intervals)):
		if intervals[q] - intervals[q - 1] > 1:
			if len(temp) > len(max_int):
				max_int = temp
			temp = [intervals[q]]
		else:
			temp.append(intervals[q])
	if len(max_int) == 0:
		max_int = temp
	return max_int


# 将事件按时间间隔划分
def get_time_intervals(event):
	event = sorted(event, key=lambda k: k['time'])      # post 按时间排序
	t0 = event[0]['time']                               # 源微博发表时间
	post_list = []
	for i in range(len(event)):
		event[i]['time'] = event[i]['time'] - t0        # 每条 post 与源博的时间距离
		if event[i]['time'] <= deadline:
			post_list.append(event[i])

	# 划分时间间隔区间
	L = post_list[-1]['time'] - post_list[0]['time']
	l = L / time_steps
	k = 0
	pre_intervals = []
	embeddings = []
	while True:
		k += 1
		index = 0
		output = []
		if L == 0:
			for post in post_list:
				output.append(post['embedding'])
			break
		start = post_list[0]['time']
		num = int(L / l)
		intset = []
		for inter in range(0, num):
			empty = 0
			interval = []
			for j in range(index, len(post_list)):
				if start <= post_list[j]['time'] < start+l:
					empty += 1
					interval.append(post_list[j]['embedding'])
				elif post_list[j]['time'] >= start+l:
					index = j-1
					break
			if empty == 0:
				output.append([])
			else:
				if post_list[-1]['time'] == start+l:
					interval.append(post_list[-1]['embedding'])
				intset.append(inter)
				output.append(interval)
			start = start + l
		con_intervals = get_continuous_intervals(intset)
		if len(pre_intervals) < len(con_intervals) < time_steps:
			l = int(0.5 * l)
			pre_intervals = con_intervals
			if l == 0:
				output = output[con_intervals[0]: con_intervals[-1]+1]
				break
		else:
			output = output[con_intervals[0]: con_intervals[-1]+1]
			break
	for item in output:
		embedding = np.zeros((300, ), dtype="float32")
		for q in range(len(item)):
			embedding += item[q]
		embedding = embedding / len(item)
		embeddings.append(embedding)
	return embeddings


# 文本分词并转换为词向量表示
def get_embeddings():
	input_data = []
	input_label = []
	event_list = get_eventlist()
	with open(weibo_embedding_path, 'r') as fe:
		embeddings = json.load(fe)

	for event in event_list:
		events = []
		label = event['label']
		event_json_path = os.path.join(weibo_json_path, event['eid'])
		with open(event_json_path, 'r') as fj:
			event_json = json.load(fj)
			for i in range(event['posts_num']):
				vec = np.zeros((300, ), dtype='float32')
				weibo = event_json[i]
				text = weibo['text']
				time = weibo['t']

				text = str_filter(text)                         # 文本过滤
				words = jieba.cut(text, cut_all=False)          # 分词
				n = 0
				for word in words:
					n += 1
					if embeddings.__contains__(word):           # 向量化
						vec = vec + np.array(embeddings[word], dtype='float32')
				if n == 0:
					vec = vec
				else:
					vec = vec / n
				post = {'embedding': vec, 'time': time}
				events.append(post)

		#print('Event length: %d' % len(Event))
		
		Embeddings = get_time_intervals(events)
		if len(Embeddings) < time_steps:
			for i in range(0, time_steps-len(Embeddings)):
				Embeddings.append([0.0] * 300)                  # 时间步补全

		input_data.append(Embeddings[: time_steps])
		input_label.append(label)

	return np.array(input_data), np.array(input_label)

'''
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('F1: %.4f Precision: %.4f Recall: %.4f'%(_val_f1, _val_precision, _val_recall))
        return
'''


def main(_):
	input_data, input_label = get_embeddings()

	# 划分训练集及测试集
	x_train = input_data[int(len(input_data)*0.2): ]
	y_train = input_label[int(len(input_label)*0.2): ]

	x_test = input_data[: int(len(input_data)*0.2)]
	y_test = input_label[: int(len(input_label)*0.2)]

	x_train = x_train.reshape(-1, time_steps, embedding_size)
	x_test = x_test.reshape(-1, time_steps, embedding_size)

	y_train = np_utils.to_categorical(y_train, num_classes=2)
	y_test = np_utils.to_categorical(y_test, num_classes=2)

	print('Data preprocessing completed----------')

	# GRU model
	model = Sequential()

	model.add(GRU(cell_num, input_shape=(time_steps, embedding_size), return_sequences=True))
	model.add(Dropout(0.5))
	model.add(GRU(cell_num, input_shape=(time_steps, embedding_size)))
	model.add(Dropout(0.5))
	model.add(Dense(output_size))
	model.add(Activation('softmax'))
	
	model.compile(optimizer=Adagrad(learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])


	print('Training------------------------------')
	history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test))

'''	
	epochs = history.epoch
	train_loss = history.history['loss']
	train_acc = history.history['binary_accuracy']

	plt.figure(0)
	plt.plot(epochs, train_loss, 'ro-')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.ylim((np.min(train_loss), np.max(train_loss)))
	plt.savefig('../result/loss.jpg')
	plt.figure(1)
	plt.plot(epochs, train_acc, 'bo-')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.ylim((np.min(train_acc), np.max(train_acc)))
	plt.savefig('../result/acc.jpg')
'''

if __name__ == '__main__':
	tf.app.run()