# encoding:utf-8

import json
import numpy as np
import tensorflow as tf
import pretreatment as pre
from gensim.models import Word2Vec

# 配置文件路径
event_txt_path = '../data/event_text/'
labels_path = '../data/label/'
vector_path = '../data/word2vec/'

# 配置参数
#time_steps = 50
#word_size = 300


# 数据预处理，返回处理后的输入数据
def get_embeddings(time_steps, embedding_size):
    model = Word2Vec.load(vector_path + 'word2vec.model')       # 读取训练好的词向量模型
    vector = model.wv                                           # vector为"word:vector"对应表

    files = pre.get_file_name(event_txt_path)                   # 获取当前目录下的所有文件名和文件数
    num_event = len(files)
    inputs_x = np.zeros((num_event, time_steps, embedding_size), dtype=np.float32)         # 初始化输入矩阵
    inputs_y = np.zeros((num_event, 2))

    with open(labels_path + 'labels.json', 'r') as fl:
        labels = json.load(fl)

    for i in range(num_event):
        inputs_y[i] = labels[files[i]]

        with open(event_txt_path + files[i], 'r', encoding='utf8') as event:
            lines = event.readlines()
            length = len(lines)                                 # 计算每个事件的评论数
            vec = np.zeros((length, 300), dtype=np.float32)     # 初始化事件矩阵
    
            for j in range(length):
                sents = lines[j].split()
                for word in sents:
                    vec[j, ] += np.array(vector[word])          # 对每条评论的各词向量求均值
                vec[j, ] = vec[j, ] / len(sents)
    
            if length < time_steps:                             # 评论数少于time_steps,补0
                inputs_x[i, 0:length, ] = vec[0:length, ]
            elif length % time_steps == 0:                      # 其他情况求均值
                m = length // time_steps
                for k in range(time_steps):
                    inputs_x[i, k, ] = np.mean(vec[k*m: k*m+m], axis=0)
            else:
                m = length // time_steps
                n = length % time_steps
                for k in range(n):
                    inputs_x[i, k, ] = np.mean(vec[k*(m+1): (k+1)*(m+1)], axis=0)
                for k in range(time_steps-n):
                    inputs_x[i, n+k, ] = np.mean(vec[((m+1)*n+k*m): ((m+1)*n+k*m+m)], axis=0)

    return num_event, inputs_x, inputs_y


def get_train_test(time_steps, embedding_size, rate=0.8):
    datas_num, inputs_x, inputs_y = get_embeddings(time_steps, embedding_size)
    train_index = int(datas_num * rate)

    train_x = np.asarray(inputs_x[: train_index], dtype=np.float32)
    train_y = np.asarray(inputs_y[: train_index], dtype=np.int32)

    test_x = np.asarray(inputs_x[train_index:], dtype=np.float32)
    test_y = np.asarray(inputs_y[train_index:], dtype=np.int32)

    return train_x, train_y, test_x, test_y


# 获取一个 batch_size 大小的数据
def get_batches(X, Y, batch_size):
    for i in range(0, len(X), batch_size):
        begin_i = i
        end_i = i + batch_size if (i + batch_size) < len(X) else len(X)
        # yield 来连续获取大小为 batch_size 的数据
        yield X[begin_i:end_i], Y[begin_i:end_i]


def BiLSTM():
    time_steps = 50                     # 序列的长度
    embedding_size = 300                # 词向量的维度
    lstm_size = 256                     # lstm层的神经元数
    batch_size = 30                     # bitch大小
    keep_prob = 0.5                     #
    epochs = 100                        # 迭代次数
    train_rate = 0.8                    # 训练样本占比

    train_x, train_y, test_x, test_y = get_train_test(time_steps, embedding_size, train_rate)

    X = tf.placeholder(tf.float32, [None, time_steps, embedding_size], name='input_x')
    Y = tf.placeholder(tf.int32, [None, 2], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')



if __name__ == '__main__':
    num, out = get_embeddings()
    print(out)
