# encoding:utf-8

import json
import os
import re


def get_file_name(file_path):						# 获取file_path路径下的文件名
	for root, dirs, files in os.walk(file_path):
		return files


def fenci(eid_path, label_path):
	char = '[+\.\!\/_,-?@$%^()#]'					# 要过滤符号列表
	events = []
	temp = []

	with open(label_path + 'label.txt') as lf:
		lines = lf.readlines()
		for line in lines:
			m = line.split()
			temp.append(m[:2])						# m[:2]为每个子事件的'eid'和'label'

	list_eid = get_file_name(eid_path)				# 文件名列表每一个文件将处理为一个events
	for eid in list_eid:
		event = [[]]
		for i in range(len(temp)):
			if temp[i][0] == str('eid:' + eid):		# 根据eid查找对应的label
				label = temp[i][1][6:]				# 忽略字符'label'
		event[0].append(eid)
		event[0].append(label)

		with open(eid_path + eid, 'r') as f:
			data = json.load(f)						# data的类型为 'list'
			event[0].append(len(data))				# len(data)表示子事件的数量
			for item in data:
				text = re.sub(char, '', item['text']).split()			# 过滤'text'里的符号并分词
				for s in text:
					if s[0:4] == 'http':
						text.remove(s)				# 过滤掉'http'开头的网址
				event.append(text)
		events.append(event)
	return events

if __name__ == '__main__':
	eid_path = '../data/event/'
	label_path = '../data/label/'

	events = fenci(eid_path, label_path)
	print(events)