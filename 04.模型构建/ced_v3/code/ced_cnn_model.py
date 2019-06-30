# encoding: utf-8

import os
import re
import json
import jieba
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv1D, Dense, GlobalMaxPool1D, GRU, Dropout, concatenate, Input
from keras.optimizers import Adam
from keras import regularizers


# 配置路径
data_save_path = '../data/Weibo/'


filter_width1 = 4			# CNN滤波器的宽度
filter_width2 = 5
filter_size = 50			# CNN滤波器的尺寸
hidden_size = 200			# 隐层单元数
batch_size = 128				
embedding_size = 300	
time_steps = 1000


def main(_):
    # 读取预处理好的数据
    print('Reading Data------------------------------')
    om_train = np.asarray(np.load(data_save_path + 'input_om_train.npy'))
    fi_train = np.asarray(np.load(data_save_path + 'input_fi_train.npy'))
    label_train = np.asarray(np.load(data_save_path + 'input_label_train.npy'))

    om_train = [[i]*time_steps for i in om_train]
    label_train = np_utils.to_categorical(label_train, num_classes=2)

    om_test = np.asarray(np.load(data_save_path + 'input_om_test.npy'))
    fi_test = np.asarray(np.load(data_save_path + 'input_fi_test.npy'))
    label_test = np.asarray(np.load(data_save_path + 'input_label_test.npy'))

    om_test = [[i]*time_steps for i in om_test]
    label_test = np_utils.to_categorical(label_test, num_classes=2)

    print('Data preprocessing completed----------')

    # CNN Layers OM
    om_input = Input(shape=(time_steps, embedding_size), name='input_om')
    om_cnn1 = Conv1D(filter_size, filter_width1, padding='same', activation='tanh')(om_input)
    om_pool1 = GlobalMaxPool1D()(om_cnn1)
    om_cnn2 = Conv1D(filter_size, filter_width2, padding='same', activation='tanh')(om_input)
    om_pool2 = GlobalMaxPool1D()(om_cnn2)

    # CNN and GRU Layers Reposts
    fi_input = Input(shape=(time_steps, embedding_size),name='input_fi')
    fi_cnn1 = Conv1D(filter_size, filter_width1, padding='same', activation='tanh')(fi_input)
    fi_cnn2 = Conv1D(filter_size, filter_width2, padding='same', activation='tanh')(fi_input)

    fi = concatenate([fi_cnn1, fi_cnn2])
    h = GRU(hidden_size, dropout=0.3)(fi)
    h_ = concatenate([om_pool1, om_pool2, h])
    re = Dropout(0.3)(h_)
    output = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(re)

    # 模型定义
    model = Model(inputs=[om_input, fi_input], outputs=output)

    # 编译模型
    model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['binary_accuracy'])

    # 模型训练
    print('Training------------------------------')
    model.fit([om_train, fi_train], label_train, epochs=1, validation_data=([om_test, fi_test], label_test), batch_size=batch_size)

    y = model.predict([om_train, fi_train], batch_size=batch_size)

    print(y)


if __name__ == "__main__":
    tf.app.run()
