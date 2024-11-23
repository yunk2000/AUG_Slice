# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 30
embedding_dim = 50
vocab_size = 5000


def get_data(corpus_path, labels_path, model):
    x = list()
    y = list()
    with open(corpus_path, 'rb') as file:
        data = pickle.load(file)
        length = len(data)
        print(length)
    for vec in data:
        result = np.array(vec).flatten()
        x.append(result)
    x = np.array(x)
    x = x.reshape((-1, max_len, embedding_dim))
    if model:
        with open(labels_path, 'rb') as file:
            y = pickle.load(file)
            y = np.array([int(x) for x in y])
    return x, y


train_corpus_path = "./data/corpus_path/train_tokens.pkl"
labels_path = "./data/tokens/train_labels.pkl"
lstm_model_path = "./data/model/lstm_model_30"
result_path = './data/result/lstm_result.txt'

x, y = get_data(train_corpus_path, labels_path, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(SpatialDropout1D(0.2, input_shape=(max_len, embedding_dim)))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_test, y_test), verbose=2)

model.save(lstm_model_path)

model = load_model(lstm_model_path)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

fwrite = open(result_path, 'a')
fwrite.write("TP:" + str(TP) + ' FP:' + str(FP) + ' FN:' + str(FN) + ' TN:' + str(TN) + '\n')

FPR = float(FP) / (FP + TN)
fwrite.write('FPR: ' + str(FPR) + '\n')
FNR = float(FN) / (TP + FN)
fwrite.write('FNR: ' + str(FNR) + '\n')

fwrite.write('Accuracy: ' + str(accuracy) + '\n')
fwrite.write('precision: ' + str(precision) + '\n')
fwrite.write('recall: ' + str(recall) + '\n')
fwrite.write('fbeta_score: ' + str(f1) + '\n')

fwrite.write('--------------------\n')
fwrite.close()
