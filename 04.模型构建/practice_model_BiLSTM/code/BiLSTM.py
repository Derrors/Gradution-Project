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
model_path = '../model/'


# 数据预处理，返回处理后的输入数据
def get_embeddings(time_steps, embedding_size):
    model = Word2Vec.load(vector_path + 'word2vec.model')       # 读取训练好的词向量模型
    vector = model.wv                                           # vector为"word:vector"对应表

    files = pre.get_file_name(event_txt_path)                   # 获取当前目录下的所有文件名和文件数
    num_event = len(files)
    inputs_x = np.zeros((num_event, time_steps, embedding_size), dtype=np.float32)         # 初始化输入矩阵
    inputs_y = np.zeros((num_event, 2), dtype=np.int32)
    seq_length = np.zeros((num_event, ), dtype=np.int32)

    with open(labels_path + 'labels.json', 'r') as fl:
        labels = json.load(fl)

    for i in range(num_event):
        eid = files[i].split('.')[0]
        inputs_y[i] = labels[eid]

        with open(event_txt_path + files[i], 'r', encoding='utf8') as event:
            lines = event.readlines()
            length = len(lines)                                 # 计算每个事件的评论数
            vec = np.zeros((length, embedding_size), dtype=np.float32)     # 初始化事件矩阵
    
            for j in range(length):
                sents = lines[j].split()
                for word in sents:
                    vec[j, ] += np.array(vector[word])          # 对每条评论的各词向量求均值
                vec[j, ] = vec[j, ] / len(sents)
    
            if length < time_steps:                             # 评论数少于time_steps,补0
                inputs_x[i, 0:length, ] = vec[0:length, ]
                seq_length[i] = length
            elif length % time_steps == 0:                      # 其他情况求均值
                seq_length[i] = time_steps
                m = length // time_steps
                for k in range(time_steps):
                    inputs_x[i, k, ] = np.mean(vec[k*m: k*m+m], axis=0)
            else:
                seq_length[i] = time_steps
                m = length // time_steps
                n = length % time_steps
                for k in range(n):
                    inputs_x[i, k, ] = np.mean(vec[k*(m+1): (k+1)*(m+1)], axis=0)
                for k in range(time_steps-n):
                    inputs_x[i, n+k, ] = np.mean(vec[((m+1)*n+k*m): ((m+1)*n+k*m+m)], axis=0)

    return inputs_x, inputs_y,  seq_length


# 根据训练比例来划分训练集和测试集
def get_train_test(time_steps, embedding_size, rate=0.8):
    inputs_x, inputs_y, seq_length = get_embeddings(time_steps, embedding_size)
    train_index = int(len(inputs_x) * rate)

    train_x = np.asarray(inputs_x[: train_index], dtype=np.float32)
    train_y = np.asarray(inputs_y[: train_index], dtype=np.int32)
    train_seq_length = np.asarray(seq_length[: train_index], dtype=np.int32)

    test_x = np.asarray(inputs_x[train_index:], dtype=np.float32)
    test_y = np.asarray(inputs_y[train_index:], dtype=np.int32)
    test_seq_length = np.asarray(seq_length[train_index:], dtype=np.int32)

    return train_x, train_y, train_seq_length, test_x, test_y, test_seq_length


def BiLSTM(inputs, seq_length, hidden_num, weights, bias):
    # 搭建双向LSTM模型
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(hidden_num),            # 前向LSTM单元
        cell_bw=tf.contrib.rnn.LSTMCell(hidden_num),            # 反向LSTM单元
        inputs=inputs,                                          # 数据输入
        sequence_length=seq_length,                             # 输入数据的有效序列长度
        dtype=tf.float32
    )
                                                                # outputs = (output_fw, output_bw)为一个turple
    outputs = tf.concat(outputs, 2)                             # 将outputs在hidden_num维度上进行拼接
    output = outputs[:, -1, :]                                  # outputs.shape = (batch_size,time_steps,hidden_num*2)

    logits = tf.matmul(output, weights) + bias                  # 取出最后时刻的输出结果进行softmax操作
    logits = tf.nn.softmax(logits)                              # logits.shape = (batch_size, classs_num)

    return logits


# 获取一个 batch_size 大小的数据
def get_batches(inputs_x, inputs_y, seq_length, batch_size):
    for i in range(0, len(inputs_x), batch_size):
        begin_i = i
        end_i = i + batch_size if (i + batch_size) < len(inputs_x) else len(inputs_x)
        # yield 来连续获取大小为 batch_size 的数据
        yield inputs_x[begin_i: end_i], inputs_y[begin_i: end_i], seq_length[begin_i: end_i]


# 计算预测结果的正确率
def compute_accuracy(pred, true):
    correct_num = 0
    for n in range(len(pred)):
        if np.argmax(pred, 1)[n] == np.argmax(true, 1)[n]:
            correct_num += 1
    return correct_num / len(pred)


def main(_):
    time_steps = 50                     # 序列的长度
    embedding_size = 300                # 词向量的维度
    hidden_num = 256                    # lstm层的神经元数
    class_num = 2                       # 类别数
    batch_size = 10                     # batch大小
    epochs = 30                         # 迭代次数
    train_rate = 0.8                    # 训练样本占比

    train_x, train_y, train_seq, test_x, test_y, test_seq = get_train_test(time_steps, embedding_size, train_rate)

    # 定义输入数据
    X = tf.placeholder(tf.float32, [None, time_steps, embedding_size], name='input_x')
    Y = tf.placeholder(tf.int32, [None, 2], name='input_y')
    seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
    # 定义并初始化全连接层的权重和偏置项
    weights = tf.Variable(tf.random_uniform([hidden_num * 2, class_num], -0.01, 0.01))
    bias = tf.Variable(tf.random_uniform([class_num], -0.01, 0.01))

    logits = BiLSTM(X, seq_length, hidden_num, weights, bias)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(             # 损失函数通过交叉熵来计算
        labels=Y,
        logits=logits,
        name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 优化方法采用梯度下降算法
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(cross_entropy_mean)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()                                # 初始化全部变量
    with tf.Session() as sess:
        sess.run(init)
        print('init variables complete...')

        for i in range(epochs):
            for train_x_batch, train_y_batch, train_seq_length in get_batches(train_x, train_y, train_seq, batch_size):
                feed_dict = {X: train_x_batch, Y: train_y_batch, seq_length: train_seq_length}
                _, train_loss, train_pred, train_true = sess.run([optimizer, cross_entropy_mean, logits, Y], feed_dict=feed_dict)
            if i % 50 == 0 and i > 0:
                train_accuracy = compute_accuracy(train_pred, train_true)
                print('step: %d train_loss: %f train_accuracy: %f' % (i, train_loss, train_accuracy))
            if i % 100 == 0 and i > 0:
                for test_x_batch, test_y_batch, test_seq_length in get_batches(test_x, test_y, test_seq, batch_size):
                    feed_dict = {X: test_x_batch, Y: test_y_batch, seq_length: test_seq_length}
                    _, test_loss, test_pred, test_true = sess.run([optimizer, cross_entropy_mean, logits, Y], feed_dict=feed_dict)
                test_accuracy = compute_accuracy(test_pred, test_true)
                print('step: %d test_loss %f test_accuracy: %f' % (i, test_loss, test_accuracy))

        saver.save(sess, model_path + 'model.bilstm')
        print('save model succeed...')


if __name__ == '__main__':
    tf.app.run()