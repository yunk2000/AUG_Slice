# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 50
max_len = 30
batch_size = 16


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
rnn_model_path = "./data/model/rnn_model_30"
result_path = './data/result/rnn_result.txt'

x, y = get_data(train_corpus_path, labels_path, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(SimpleRNN(32, input_shape=(max_len, embedding_dim)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=batch_size,
                    validation_split=0.2)

model.save(rnn_model_path)

model = load_model(rnn_model_path)

test_loss, test_acc = model.evaluate(x_test, y_test)

y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.7).astype("int32")

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

fwrite.write('Accuracy: ' + str(test_acc) + '\n')
fwrite.write('precision: ' + str(precision) + '\n')
fwrite.write('recall: ' + str(recall) + '\n')
fwrite.write('fbeta_score: ' + str(f1) + '\n')

fwrite.write('--------------------\n')
fwrite.close()
