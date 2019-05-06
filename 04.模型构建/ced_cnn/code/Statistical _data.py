# encoding: utf-8

import os
import re
import json
import jieba
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# 配置路径
weibo_embedding_path = '../data/embedding/weibo.json'
weibo_json_path = '../data/Weibo/json/'
weibo_label_path = '../data/Weibo/label.txt'

n = 140						# 文本最大长度
N = 10						# 转发聚合度
L = 20001					# 最大转发数量（包含源微博）


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
def main(_):
    count = []
    word_count = 0
    post_count = []
    event_list = get_eventlist()
    word_dict = {}


    for event in event_list:
        event_json_path = os.path.join(weibo_json_path, event['eid'])
        with open(event_json_path, 'r') as fj:
            event_json = json.load(fj)
            post_count.append(event['posts_num'])
            for i in range(0, event['posts_num'], 10):
                num = 0
                for q in range(10):
                    if i+q < event['posts_num']:
                        weibo = event_json[i+q]
                        text = weibo['text']
                        text = str_filter(text)                         # 文本过滤
                        words = jieba.cut(text, cut_all=False)          # 分词
                        for word in words:
                            num += 1
                            if not word_dict.__contains__(word):
                                word_count += 1
                                word_dict['word'] = word_count
                count.append(num)
    
    data = np.asarray(count)
    plt.figure(0)
    plt.hist(data, bins=5, facecolor='blue', edgecolor='black', bottom=1, alpha=0.7)
    plt.xlabel('Words count')
    plt.ylabel('Nums')
    plt.title('Statistical data')
    plt.savefig('词数分布统计.jpg')

    print('最大词数：' + str(np.max(count)))
    print('最小词数：' + str(np.min(count)))
    print('词表大小：' + str(word_count))

    data2 = np.asarray(post_count)
    plt.figure(1)
    plt.hist(data2, bins=10, facecolor='blue', edgecolor='black', bottom=1, alpha=0.7)
    plt.xlabel('Post count')
    plt.ylabel('Nums')
    plt.title('Statistical data')
    plt.savefig('转发数量分布统计.jpg')


if __name__ == "__main__":
    tf.app.run()