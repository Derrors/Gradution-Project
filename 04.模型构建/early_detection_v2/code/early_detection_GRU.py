# encoding:utf-8

import os
import re
import json
import jieba
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, GRU, Dropout
from keras.optimizers import Adam, Adagrad
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# 配置路径信息
weibo_json_path = '../../Weibo/json/'
weibo_label_path = '../../Weibo/label.txt'
weibo_embedding_path = '../data/embedding/weibo.json'

# 配置参数信息
tf.app.flags.DEFINE_integer('N', 20, "TimeSteps")
tf.app.flags.DEFINE_integer('Hours', 10, "Deadline")			
FLAGS = tf.app.flags.FLAGS

TimeSteps = FLAGS.N 					# 时间步数
Deadline = FLAGS.Hours * 3600			# Deadline
EmbeddingSize = 300
BatchSize = 30
OutputSize = 2                          # 输出维度
CellNum = 125                           # 隐层单元数
LR = 0.001                              # 学习率


# 过滤文本信息中的非中文字符
def StrFilter(string):
	chars = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
	string = ''.join(string.split())
	return re.sub(chars, '', string)


# 获取事件列表，格式为：eid, label, posts_num
def GetEventList():
	EventList = []
	with open(weibo_label_path, 'r') as fl:
		lines = fl.readlines()
		for line in lines:
			event_str = line.split()
			eid = (event_str[0].split(':'))[1]
			label = (event_str[1].split(':'))[1]
			posts_num = len(event_str[2:])
			event = {'eid': eid, 'label': label, 'posts_num': posts_num}
			EventList.append(event)
	return EventList


# 获取最大时间跨度的连续时间间隔
def GetContinuousIntervals(intervals):
	maxInt = []
	temp = [intervals[0]]
	for q in range(1, len(intervals)):
		if intervals[q] - intervals[q - 1] > 1:
			if len(temp) > len(maxInt):
				maxInt = temp
			temp = [intervals[q]]
		else:
			temp.append(intervals[q])
	if len(maxInt) == 0:
		maxInt = temp
	return maxInt


# 将事件按时间间隔划分
def GetTimeIntervals(Event):
	Event = sorted(Event, key=lambda k: k['time'])      # post 按时间排序
	t0 = Event[0]['time']                               # 源微博发表时间
	PostList = []
	for i in range(len(Event)):
		Event[i]['time'] = Event[i]['time'] - t0        # 每条 post 与源博的时间距离
		if Event[i]['time'] <= Deadline:
			PostList.append(Event[i])

	# 划分时间间隔区间
	L = PostList[-1]['time'] - PostList[0]['time']
	l = L / TimeSteps
	k = 0
	PreIntervals = []
	Embeddings = []
	while True:
		k += 1
		index = 0
		output = []
		if L == 0:
			for post in PostList:
				output.append(post['embedding'])
			break
		start = PostList[0]['time']
		num = int(L / l)
		intset = []
		for inter in range(0, num):
			empty = 0
			interval = []
			for j in range(index, len(PostList)):
				if start <= PostList[j]['time'] < start+l:
					empty += 1
					interval.append(PostList[j]['embedding'])
				elif PostList[j]['time'] >= start+l:
					index = j-1
					break
			if empty == 0:
				output.append([])
			else:
				if PostList[-1]['time'] == start+l:
					interval.append(PostList[-1]['embedding'])
				intset.append(inter)
				output.append(interval)
			start = start+l
		ConIntervals = GetContinuousIntervals(intset)
		if len(PreIntervals) < len(ConIntervals) < TimeSteps:
			l = int(0.5 * l)
			PreIntervals = ConIntervals
			if l == 0:
				output = output[ConIntervals[0]: ConIntervals[-1]+1]
				break
		else:
			output = output[ConIntervals[0]: ConIntervals[-1]+1]
			break
	for item in output:
		embedding = np.zeros((300, ), dtype=np.float32)
		for q in range(len(item)):
			embedding += item[q]
		embedding = embedding / len(item)
		Embeddings.append(embedding)
	return Embeddings


# 文本分词并转换为词向量表示
def GetEmbeddings():
	InputData = []
	InputLabel = []
	EventList = GetEventList()
	with open(weibo_embedding_path, 'r') as fe:
		embeddings = json.load(fe)

	for event in EventList:
		Event = []
		label = event['label']
		event_json_path = os.path.join(weibo_json_path, event['eid'])
		with open(event_json_path, 'r') as fj:
			event_json = json.load(fj)
			for i in range(event['posts_num']):
				vec = np.zeros((300, ), dtype=np.float32)
				weibo = event_json[i]
				text = weibo['text']
				time = weibo['t']

				text = StrFilter(text)                          # 文本过滤
				words = jieba.cut(text, cut_all=False)          # 分词
				n = 0
				for word in words:
					n += 1
					if embeddings.__contains__(word):           # 向量化
						vec = vec + np.array(embeddings[word], dtype=np.float32)
				if n == 0:
					vec = vec
				else:
					vec = vec / n
				post = {'embedding': vec, 'time': time}
				Event.append(post)
		#print('Event length: %d' % len(Event))
		
		Embeddings = GetTimeIntervals(Event)
		if len(Embeddings) < TimeSteps:
			for i in range(0, TimeSteps-len(Embeddings)):
				Embeddings.append([0.0] * 300)                  # 时间步补全

		InputData.append(Embeddings[: TimeSteps])
		InputLabel.append(label)

	return np.array(InputData), np.array(InputLabel)


def main(_):
	InputData, InputLabel = GetEmbeddings()

	# 划分训练集及测试集
	x_train = InputData[: int(len(InputData)*0.8)]
	y_train = InputLabel[: int(len(InputLabel)*0.8)]

	x_test = InputData[int(len(InputData)*0.8):]
	y_test = InputLabel[int(len(InputLabel)*0.8):]

	x_train = x_train.reshape(-1, TimeSteps, EmbeddingSize)
	X_test = x_test.reshape(-1, TimeSteps, EmbeddingSize)
	y_train = np_utils.to_categorical(y_train, num_classes=2)
	y_test = np_utils.to_categorical(y_test, num_classes=2)

	print('Data preprocessing completed----------')

	# GRU model
	model = Sequential()

	model.add(GRU(CellNum, input_shape=(TimeSteps, EmbeddingSize)))
	# model.add(Dropout(0.3))
	model.add(Dense(OutputSize))
	# model.add(Dropout(0.3))
	model.add(Activation('softmax'))

	model.compile(optimizer=Adagrad(LR), loss='binary_crossentropy', metrics=['binary_accuracy'])

	print('Training------------------------------')
	history = model.fit(x_train, y_train, epochs=30, batch_size=BatchSize)
	epochs = history.epoch
	train_loss = history.history['loss']
	train_acc = history.history['binary_accuracy']

	print('Testing------------------------------')
	loss_and_accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=0)
	print('test loss: ', loss_and_accuracy[0])
	print('test accuracy: ', loss_and_accuracy[1])

	plt.figure(0)

	plt.subplot(211)
	plt.plot(epochs, train_loss, 'rv-')
	plt.ylabel('Loss')
	plt.ylim((np.min(train_loss), np.max(train_loss)))
	plt.title('Deadline = %d hours' % (FLAGS.Hours))
	
	plt.subplot(212)
	plt.plot(epochs, train_acc, 'b^-')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.ylim((np.min(train_acc), np.max(train_acc)))

	save_path = os.path.join('../result/', str('deadline_') + str(FLAGS.Hours) + '_hours.jpg')
	plt.savefig(save_path)


if __name__ == '__main__':
	tf.app.run()