# encoding: utf-8

import os
import re
import json
import jieba
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPooling1D, GRU
from keras.layers import Activation, concatenate, Input
from keras.optimizers import Adam, Adagrad


# 配置路径
weibo_embedding_path = '../data/embedding/weibo.json'
weibo_json_path = '../../Weibo/json/'
weibo_label_path = '../../Weibo/label.txt'


# 参数定义
tf.app.flags.DEFINE_float('alpha', 0.975, 'threshold')		# 概率阈值 α
FLAGS = tf.app.flags.FLAGS

n = 140						# 文本最大长度
N = 10						# 转发聚合度
L = 20001					# 最大转发数量（包含源微博）
filter_width1 = 4			# CNN滤波器的宽度
filter_width2 = 5
filter_size = 50			# CNN滤波器的尺寸
hidden_size = 100			# 隐层单元数
batch_size = 20				
embedding_size = 300	
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


# 将文本分词并转换为词向量表示
def get_word_vector():
	input_om = []
	input_fi = []
	input_label = []

	event_list = get_eventlist()
	with open(weibo_embedding_path, 'r') as fe:
		weibo_embeddings = json.load(fe)

	for event in event_list:
		om = []
		fi = []
		label = event['label']
		event_json_path = os.path.join(weibo_json_path, event['eid'])

		with open(event_json_path, 'r') as fj:
			event_json = json.load(fj)
			reposts = []
			for i in range(event['posts_num']):
				w = []
				weibo = event_json[i]
				text = weibo['text']

				text = str_filter(text)                         # 文本过滤
				words = jieba.cut(text, cut_all=False)          # 分词
				l = 0

				# 转换为词向量并拼接
				for word in words:
					if l < n:
						if i == 0:
							if weibo_embeddings.__contains__(word):
								om.append(np.array(weibo_embeddings[word], dtype='float32'))
							else:
								om.append(np.zeros((embedding_size, ), dtype='float32'))
								l += 1
						else:
							if weibo_embeddings.__contains__(word):
								w.append(np.array(weibo_embeddings[word], dtype='float32'))
							else:
								w.append(np.zeros((embedding_size, ), dtype='float32'))
								l += 1
					else:
						break

				# 填充补长（句子）
				for k in range(n-l):
					if i == 0:
						om.append(np.zeros((embedding_size, ), dtype='float32'))
					else:
						w.append(np.zeros((embedding_size, ), dtype='float32'))
				if i != 0:
					reposts.append(w)

			# 填充补长（事件）
			if event['posts_num'] < L:
				for i in range(L - event['posts_num']):
					reposts.append(np.zeros((n*embedding_size, ), dtype='float32'))

			# 划分时间步并拼接
			for j in range(0, L, 10):
				temp = []
				for q in range(N):
					temp.extend(reposts[j+q])
				fi.append(temp)

		input_om.append(om)
		input_fi.append(fi)
		input_label.append(label)

	return np.array(input_om), np.array(input_fi), np.array(input_label)


def main(_):
	input_om, input_fi, input_label = get_word_vector()

	# 划分训练集与测试集
	om_train = input_om[: int(0.75*len(input_om))]
	om_test = input_om[int(0.75*len(input_om)): ]

	fi_train = input_fi[: int(0.75*len(input_fi))]
	fi_test = input_fi[int(0.75*len(input_fi)): ]

	label_train = input_label[: int(0.75*len(input_label))]
	label_test = input_label[int(0.75*len(input_label)): ]

	# 类别标签格式转换
	label_train = np_utils.to_categorical(label_train, num_classes=2)
	label_test = np_utils.to_categorical(label_test, num_classes=2)

	print('Data preprocessing completed----------')

	# CNN Layers OM
	om_input = Input(shape=(n*embedding_size, ))
	om_cnn1 = Conv1D(filter_size, filter_width1, activation='relu')(om_input)
	om_cnn2 = Conv1D(filter_size, filter_width2, activation='relu')(om_cnn1)
	om_pooling = MaxPooling1D()(om_cnn2)
	r = Dense(hidden_size)(om_pooling)

	# CNN and GRU Layers Reposts
	fi_input = Input(shape=(2000, N*n*embedding_size))
	fi_cnn1 = Conv1D(filter_size, filter_width1, activation='relu')(fi_input)
	fi_cnn2 = Conv1D(filter_size, filter_width2, activation='relu')(fi_cnn1)
	fi_pooling = MaxPooling1D()(fi_cnn2)
	gru_input = Dense(hidden_size)(fi_pooling)
	h = GRU(hidden_size)(gru_input)

	# OM 与 Repsots 的输出拼接
	merge = concatenate([r, h])
	output = Dense(2)((merge))
	prediction = Activation('softmax')(output)
	
	# 模型定义
	model = Model(inputs=[om_input, fi_input], output=prediction)



	# 目标函数定义


	# 编译模型



if __name__ == "__main__":
	tf.app.run()
