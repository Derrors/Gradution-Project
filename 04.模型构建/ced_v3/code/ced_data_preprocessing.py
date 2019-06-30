# encoding: utf-8

import os
import re
import gc
import math
import json
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# 配置路径
weibo_json_path = '../data/Weibo/json/'
weibo_label_path = '../data/Weibo/label.txt'
data_save_path = '/home/qhli/CED/ced_cnn/data/'


# 参数定义
N = 10						# 转发聚合度			
vocabulary_size = 1000
time_steps = 1000

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
	
	fi_train = input_fi[: int(0.75*len(input_fi))]
	seq_train = sequence_length[: int(0.75*len(input_fi))]
	label_train = input_label[: int(0.75*len(input_label))]

	np.save(data_save_path + 'input_fi_train.npy', fi_train)
	np.save(data_save_path + 'input_label_train.npy', label_train)
	np.save(data_save_path + 'sequence_length_train.npy', seq_train)

	del fi_train, label_train, seq_train
	gc.collect()

	fi_test = input_fi[int(0.75*len(input_fi)): ]
	seq_test = sequence_length[int(0.75*len(input_fi)): ]
	label_test = input_label[int(0.75*len(input_fi)): ]

	np.save(data_save_path + 'input_fi_test.npy', fi_test)
	np.save(data_save_path + 'input_label_test.npy', label_test)
	np.save(data_save_path + 'sequence_length_test.npy', seq_test)

	del input_fi, input_label, sequence_length, fi_test, label_test, seq_test
	gc.collect()
	
	print('Data preprocessing completed.')


if __name__ == "__main__":
	get_word_vector()