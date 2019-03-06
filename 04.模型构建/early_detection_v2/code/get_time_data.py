# encoding:utf-8

import os
import json
from xlsxwriter import *

weibo_json_path = '../../Weibo/json/'
weibo_label_path = '../../Weibo/label.txt'
excel_save_path = '../data/time_data/time_data.xlsx'
txt_save_path = '../data/time_data/time_data.txt'


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


def GetExcel():
	EventList = GetEventList()

	workbook = Workbook(excel_save_path)
	id_label_posts = workbook.add_worksheet('id_label_posts')
	id_label_times = workbook.add_worksheet('id_label_times')

	cell_format = workbook.add_format({'align': 'center', 'valign': 'center'})
	num_format = workbook.add_format({'align': 'center', 'valign': 'center', 'num_format': '0.00'})

	id_label_posts.write(0, 0, 'id', cell_format)
	id_label_posts.write(0, 1, 'label', cell_format)
	id_label_posts.write(0, 2, 'posts_num', cell_format)

	id_label_times.write(0, 0, 'id', cell_format)
	id_label_times.write(0, 1, 'label', cell_format)
	id_label_times.write(0, 2, 'times', cell_format)

	i = 1
	for event in EventList:
		label = event['label']
		posts_num = event['posts_num']

		id_label_posts.write(i, 0, i, cell_format)
		id_label_posts.write(i, 1, label, cell_format)
		id_label_posts.write(i, 2, posts_num, cell_format)

		id_label_times.write(i, 0, i, cell_format)
		id_label_times.write(i, 1, label, cell_format)

		j = 2
		event_json_path = os.path.join(weibo_json_path, event['eid'])
		with open(event_json_path, 'r') as fj:
			event_json = json.load(fj)

			t0 = event_json[0]['t']
			for k in range(event['posts_num']):
				post = event_json[k]
				time = (post['t'] - t0) / 60
				if time <= 6000
				id_label_times.write(i, j, time, num_format)
				j += 1
		i += 1
	workbook.close()


def GetTxt():
	EventList = GetEventList()

	ft = open(txt_save_path, 'w')

	for event in EventList:
		posts_num = event['posts_num']
		event_json_path = os.path.join(weibo_json_path, event['eid'])
		with open(event_json_path, 'r') as fj:
			event_json = json.load(fj)

			t0 = event_json[0]['t']
			for k in range(event['posts_num']):
				post = event_json[k]
				time = (post['t'] - t0) / 60
				ft.write(str(time))
				ft.write(' ')
			ft.write('/n')

	ft.close()


if __name__ == '__main__':
	GetTxt()	
