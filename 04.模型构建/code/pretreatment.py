# encoding:utf-8

import json
import os
import re
from gensim.models import word2vec

# 配置文件路径
eid_path = '../data/event/'							# 源文件路径
label_path = '../data/label/'						# 标签路径
event_path = '../data/event_text/'					# 分词处理后的保存路径
vector_path = '../data/word2vec/'					# 词向量保存路径


def get_file_name(file_path):						# 获取file_path路径下的文件名
	for root, dirs, files in os.walk(file_path):
		return files


def fenci(eid_path, label_path, event_path):
	char = '[+\.\!\/_,-?\[\]@*&""\'‘’”“$%„{}^()#]'	# 要过滤符号列表

	labels = {}

	fl = open(label_path + 'labels.json', 'w')		# 输出标签文件
	# fl = open(label_path + 'labels.txt', 'w')
	with open(label_path + 'label.txt') as lf:
		lines = lf.readlines()
		for line in lines:
			m = line.split()						# m[:2]为每个子事件的'eid'和'label'
			m = m[:2]
			eid = m[0].split(':')[1]
			label = m[1].split(':')[1]
			if int(label) == 0:						# 两类问题的第一类：否
				labels[eid] = [1, 0]
			else:
				labels[eid] = [0, 1]				# 两类问题的第一类：是
			# fl.write(eid + ' ' + label + '\n')		# 输出格式'eid label'
	json.dump(labels, fl)
	fl.close()

	list_eid = get_file_name(eid_path)				# 文件名列表
	for eid in list_eid:
		fe = open(event_path + eid + '.txt', 'w', encoding='utf8')		# 将每个event的分词结果输出

		with open(eid_path + eid, 'r') as f:
			data = json.load(f)						# data的类型为 'list'
			for item in data:
				text = re.sub(char, '', item['text']).split()			# 过滤'text'里的符号并分词
				for s in text:
					if s[0:4] == 'http':			# 过滤掉'http'开头的网址
						text.remove(s)
				for token in text:
					fe.write(token + ' ')
				fe.write('\n')
		fe.close()


def word2vector(word_path, vector_path):
	file_list = get_file_name(word_path)
	i = int(0)
	for file in file_list:
		sents = word2vec.Text8Corpus(word_path + file)					# 读取词文件

		if i == 0:														#  第一次训练，则建立模型
			# sents:词列表, size:词向量维数, window:窗口大小, min_count:忽略词频少的词, worker:线程数
			# sg=1, hs=0, negative=5:采用5个负样本的skip-gram算法
			model = word2vec.Word2Vec(sents, size=300, window=5, min_count=1, workers=4, sg=1, hs=0, negative=5)
			i += 1
		else:															# 不是第一次训练，则增量训练
			model.build_vocab(sents, update=True)						# 更新词表
			model.train(sents, total_examples=model.corpus_count, epochs=model.iter)
			i += 1
		print(file + ' 训练成功...')
	model.save(vector_path + 'word2vec.model')


if __name__ == '__main__':
	fenci(eid_path, label_path, event_path)
	word2vector(event_path, vector_path)