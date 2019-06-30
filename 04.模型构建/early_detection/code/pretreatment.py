# encoding:utf-8

import json
import os
import re
import time
import numpy as np

# 配置文件路径
eid_path = '../../Twitter/json/'						# 源文件路径
label_path_read = '../../Twitter/'					# 标签路径
label_path_write = '../label/'
event_path = '../event_text/'					    # 分词处理后的保存路径
time_path = '../event_time/'
index_path = '../event_index/'
word2vec_path_read = '../../../word2vector/'			# 词向量读取路径
word2vec_path_write = '../word2vec/'
embedding_size = 300


def get_file_name(file_path):						# 获取file_path路径下的文件名
	for root, dirs, files in os.walk(file_path):
		return files


def get_time(date):
	dt_time = time.strptime(date, '%a %b %d %H:%M:%S %z %Y')
	dt_sec = time.mktime(dt_time)
	return dt_sec


def fenci(eid_path, label_path_read, label_path_write, event_path):
	char = '[+\.\!\/_,-?\[\]@*&""\'‘’”“$%„{}^()#]'	# 要过滤符号列表

	labels = {}

	fl = open(label_path_write + 'labels.json', 'w')		# 输出标签文件
	# fl = open(label_path + 'labels.txt', 'w')
	with open(label_path_read + 'label.txt', 'r') as lf:
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
			# fl.write(eid + ' ' + label + '\n')	# 输出格式'eid label'
	json.dump(labels, fl)
	fl.close()

	list_eid = get_file_name(eid_path)				# 文件名列表
	for eid in list_eid:
		fe = open(event_path + eid + '.txt', 'w', encoding='utf8')		# 将每个event的分词结果输出
		ft = open(time_path + eid + '.txt', 'w')

		with open(eid_path + eid, 'r') as f:
			data = json.load(f)						# data的类型为 'list'
			n = len(data)
			t0 = get_time(data[0]['created_at'])

			for item in data:

				t = get_time(item['created_at'])
				ft.write(str(int(t-t0)))

				text = re.sub(char, '', item['text']).split()			# 过滤'text'里的符号并分词
				for s in text:
					if s[0:4] == 'http':			# 过滤掉'http'开头的网址
						text.remove(s)
				for token in text:
					fe.write(token + ' ')

				n -= 1
				if n > 0:
					fe.write('\n')
					ft.write('\n')
		fe.close()
		ft.close()


def build_embeddings(word2vec_path_read, word2vec_path_write):
	words_embeddings = {}
	with open(word2vec_path_read + 'Google_IJCAI2016_w2v.txt', 'r') as files:
		lines = files.readlines()
		for line in lines:
			l = line.split()
			word = l[0]
			value = l[1:]
			words_embeddings[word] = value

	with open(word2vec_path_write + 'Google_IJCAI2016_w2v.json', 'w') as fj:
		json.dump(words_embeddings, fj)

	print('word-vector build succeed...')


# 数据预处理，返回处理后的输入数据
def get_embeddings(time_path, index_path, embedding_size):
	with open(word2vec_path_write + 'Google_IJCAI2016_w2v.json', 'r') as fj:
		vector = json.load(fj)

	files = get_file_name(event_path)                   # 获取当前目录下的所有文件名和文件数
	num_event = len(files)

	for i in range(num_event):
		eid = files[i].split('.')[0]
		ft = open(time_path + eid + '.txt', 'r')
		time_lines = ft.readlines()
		with open(event_path + files[i], 'r', encoding='utf8') as event:
			lines = event.readlines()
			length = len(lines)                                 # 计算每个事件的评论数
			vec = np.zeros((length, embedding_size+1), dtype=np.float32)     # 初始化事件矩阵

			for j in range(length):
				vec[j, 0] = int(time_lines[j])
				sents = lines[j].split()
				for word in sents:
					if vector.__contains__(word):
						vec[j, 1: ] = vec[j, 1: ] + np.array(vector[word], dtype=np.float32)     # 对每条评论的各词向量求均值
				if len(sents) == 0:
					vec[j, 1: ] = vec[j, 1: ]
				else:
					vec[j, 1: ] = vec[j, 1: ] / len(sents)
			ft.close()
			with open(index_path + files[i], 'w', encoding='utf8') as fi:
				for k in range(length):
					for n in range(embedding_size+1):
						fi.write(str(vec[k, n]))
						if n < embedding_size:
							fi.write(' ')
					if k < length-1:
						fi.write('\n')


if __name__ == '__main__':
	fenci(eid_path, label_path_read, label_path_write, event_path)
	build_embeddings(word2vec_path_read, word2vec_path_write)
	get_embeddings(time_path, index_path, embedding_size)
