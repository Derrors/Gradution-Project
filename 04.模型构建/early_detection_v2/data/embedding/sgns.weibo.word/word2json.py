# encoding:utf-8
import json

dic = {}
with open('sgns.weibo.word', 'r', encoding='utf8') as f:
	lines = f.readlines()
	for line in lines:
		word = (line.split())[0]
		embedding = (line.split())[1:]
		embeddings = [float(i) for i in embedding]
		dic[word] = embeddings

with open('weibo.json', 'w') as fj:
	json.dump(dic, fj)

print('word-vector build succeed...')