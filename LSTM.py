import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import nltk
from nltk.corpus import stopwords
import numpy as np


def convert_train():
    global train_label
    global train_comments
    global train_sequence
    # 读取训练集
    with open('train2.csv', 'r', encoding='utf8') as file:
        line = csv.reader(file)
        first = 1
        for row in line:
            if not first:
                train_comments.append(row[1])
                if row[2] == 'positive':
                    train_label.append(1)
                else:
                    train_label.append(0)
            first = 0
    # 去除停用词和标点符号,存进train_sequence
    stop_words = set(stopwords.words('english'))
    for comment in train_comments:
        words = nltk.word_tokenize(comment)
        line = []
        for word in words:
            if word.isalpha() and word not in stop_words:
                line.append(word)
        train_sequence.append(line)


def convert_test():
    global test_comments
    global test_sequence
    with open('test_data.csv', 'r', encoding='utf8') as file:
        line = csv.reader(file)
        first = 1
        for row in line:
            if not first:
                test_comments.append(row[1])
            first = 0
    # 去除停用词和标点符号,存进test_sequence
    stop_words = set(stopwords.words('english'))
    for comment in test_comments:
        words = nltk.word_tokenize(comment)
        line = []
        for word in words:
            if word.isalpha() and word not in stop_words:
                line.append(word)
        test_sequence.append(line)


def build_lstm(batch_size, epochs, hidden_feature, dropout):
    global classifier
    global max_words
    classifier.add(Embedding(89483, 256, input_length=max_words))
    classifier.add(LSTM(hidden_feature, dropout=dropout))
    classifier.add(Dense(1))
    classifier.add(Activation('sigmoid'))
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier.fit(train_sequence, train_label, batch_size=batch_size, epochs=epochs)


def print_result():
    global test_label
    with open('submission.csv', 'r', encoding='utf8') as file:
        line = csv.reader(file)
        with open('submission2.csv', 'w', encoding='utf8', newline='') as new_file:
            newline = csv.writer(new_file)
            first = 1
            number = 0
            for row in line:
                if not first:
                    if test_label[number] == 1:
                        row[-1] = 'positive'
                    else:
                        row[-1] = 'negative'
                    number += 1
                    newline.writerow(row)
                else:
                    newline.writerow(row)
                first = 0


if __name__ == '__main__':
    max_words = 850  # 句子最多单词数为850
    # 训练集
    train_comments = []
    train_label = []
    train_sequence = []
    convert_train()  # 读取训练集评论句子读入train_comments，标签读入train_label,并将句子单词化，存进train_sequence
    tokenizer = Tokenizer(num_words=max_words)    # 实例化，句子最多单词数为max_words
    tokenizer.fit_on_texts(train_sequence)  # 构建单词索引结构
    word_index = tokenizer.word_index       # 获取词索引
    train_sequence = tokenizer.texts_to_sequences(train_sequence)  # 将单词转化为数字
    train_sequence = sequence.pad_sequences(train_sequence, maxlen=max_words)  # 此处设置每个句子最长不超过max_words
    train_sequence = np.array(train_sequence)   # 将list转化为numpy
    train_label = np.array(train_label)
    # 测试集
    test_comments = []
    test_label = []
    test_sequence = []
    convert_test()   # 读取测试集评论句子读入test_comments，,并将句子单词化，存进test_sequence
    tokenizer = Tokenizer(num_words=max_words)    # 实例化，句子最多单词数为max_words
    tokenizer.fit_on_texts(test_sequence)   # 构建单词索引结构
    word_index = tokenizer.word_index       # 获取词索引
    test_sequence = tokenizer.texts_to_sequences(test_sequence)    # 将单词转化为数字
    test_sequence = sequence.pad_sequences(test_sequence, maxlen=max_words)   # 此处设置每个句子最长不超过max_words
    test_sequence = np.array(test_sequence)     # 将list转化为numpy
    test_label = np.array(test_label)
    # 构建LSTM
    classifier = Sequential()   # 实例化
    build_lstm(32, 10, 128, 0.2)  # 参数按顺序分别代表batch_size, epochs, hidden_feature, dropout
    # 预测测试集
    test_label = classifier.predict_classes(test_sequence)
    print_result()  # 打印结果到submission1.csv
