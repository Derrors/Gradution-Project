# encoding: utf-8

import os
import re
import json
import jieba
import numpy as np


# 配置路径
weibo_embedding_path = '../data/embedding/weibo.json'
weibo_json_path = '../data/Weibo/json/'
weibo_label_path = '../data/Weibo/label.txt'
data_save_path = '../data/Weibo/'


# 参数定义
N = 10						# 转发聚合度
L = 10001					# 最大转发数量（包含源微博）	
embedding_size = 300	


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

	print('Data preprocessing is starting----------')

	for event in event_list:
		fi = []
		input_label.append(event['label'])
		event_json_path = os.path.join(weibo_json_path, event['eid'])

		fj = open(event_json_path, 'r')
		event_json = json.load(fj)
		reposts = []
		for i in range(event['posts_num']):
			vec = np.zeros((embedding_size, ), dtype='float32')
			weibo = event_json[i]
			text = weibo['text']

			text = str_filter(text)                         # 文本过滤
			words = jieba.cut(text, cut_all=False)          # 分词
				
			# 转换为词向量并拼接
			n = 0
			for word in words:
				if weibo_embeddings.__contains__(word):
					vec += np.array(weibo_embeddings[word], dtype='float32')
					n += 1

			if n > 0:
				vec = vec / n

			if i == 0:
				input_om.append(vec)
			else:
				reposts.append(vec)
		
		fj.close()

		# 填充补长（事件）
		if event['posts_num'] < L:
			for i in range(L - event['posts_num']):
				reposts.append(np.zeros((embedding_size, ), dtype='float32'))

		# 划分时间步并拼接
		for j in range(0, L-1, 10):
			fi.append(np.mean(reposts[j: j+N], axis=0))

		input_fi.append(fi)

	print(np.shape(input_om))
	print(np.shape(input_fi))
	print(np.shape(input_label))

	np.save(data_save_path + 'input_om_test.npy', input_om[: int(0.2*len(input_fi))])
	np.save(data_save_path + 'input_fi_test.npy', input_fi[: int(0.2*len(input_fi))])
	np.save(data_save_path + 'input_label_test.npy', input_label[: int(0.2*len(input_fi))])
	
	np.save(data_save_path + 'input_om_train.npy', input_om[int(0.2*len(input_fi)): ])
	np.save(data_save_path + 'input_fi_train.npy', input_fi[int(0.2*len(input_fi)): ])
	np.save(data_save_path + 'input_label_train.npy', input_label[int(0.2*len(input_fi)): ])
	
	print('Data preprocessing completed---------')


if __name__ == "__main__":
	get_word_vector()
