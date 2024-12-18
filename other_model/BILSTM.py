# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import random
import tensorflow as tf
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_features = 50
units = 64
maxlen = 30
epochs = 20
batch_size = 16
verbose = 1
patience = 3

callbacks = [EarlyStopping('val_acc', patience=patience)]

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

val_acc = 0
val_epoch = 0


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
    x = x.reshape(-1, 30, 50)
    if model:
        with open(labels_path, 'rb') as file:
            y = pickle.load(file)
            y = np.array([int(x) for x in y])
    return x, y


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def train(corpus_path, labels_path, biLstm_model_path, result_path):
    x, y = get_data(corpus_path, labels_path, 1)
    input_dim = x.shape[1]
    print(len(x))

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=seed)

    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(maxlen, max_features)))

    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc']
                  )

    history = model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, validation_data=(xtest, ytest))

    model.save(biLstm_model_path)

    model = load_model(biLstm_model_path)
    predictions = model.predict(xtest)

    threshold = 0.5
    predicted_labels = (predictions >= threshold).astype(int)
    pred = []
    for i in predicted_labels:
        pred.append(i[0])
    a0 = 0
    a1 = 0
    for i in pred:
        if i == 0:
            a0 = a0 + 1
        else:
            a1 = a1 + 1
    print(a0, a1)
    predicted_labels = np.array(pred)
    accuracy = np.mean(predicted_labels == ytest)

    TP = np.sum((predicted_labels == 1) & (ytest == 1))
    FP = np.sum((predicted_labels == 1) & (ytest == 0))
    TN = np.sum((predicted_labels == 0) & (ytest == 0))
    FN = np.sum((predicted_labels == 0) & (ytest == 1))
    print(TP)
    print(FP)
    print(TN)
    print(FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + FP + TN + FN)

    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1_score)
    print("Accuracy:", accuracy)
    print(TP + FP + TN + FN)

    fwrite = open(result_path, 'a')
    fwrite.write("TP:" + str(TP) + ' FP:' + str(FP) + ' FN:' + str(FN) + ' TN:' + str(TN) + '\n')

    FPR = float(FP) / (FP + TN)
    fwrite.write('FPR: ' + str(FPR) + '\n')
    FNR = float(FN) / (TP + FN)
    fwrite.write('FNR: ' + str(FNR) + '\n')

    fwrite.write('Accuracy: ' + str(accuracy) + '\n')
    fwrite.write('precision: ' + str(precision) + '\n')
    fwrite.write('recall: ' + str(recall) + '\n')
    fwrite.write('fbeta_score: ' + str(f1_score) + '\n')

    fwrite.write('--------------------\n')
    fwrite.close()


def atest(test_corpus_path, bgru_model_path):
    x = get_data(test_corpus_path, " ", 0)
    print(x)
    model = load_model(bgru_model_path)
    predictions = model.predict(x)
    threshold = 0.5
    predicted_labels = (predictions >= threshold).astype(int)
    num0 = 0
    num1 = 1
    for i in predicted_labels:
        if i == 0:
            num0 = num0 + 1
        else:
            num1 = num1 + 1
    print(num0, num1, "************")

    pred = []
    for i in predicted_labels:
        pred.append(i[0])
    predicted_labels = np.array(pred).reshape(-1, 1)
    return predictions, predicted_labels


def main():
    train_corpus_path = "./data/corpus_path/train_tokens.pkl"
    labels_path = "./data/tokens/train_labels.pkl"
    biLstm_model_path = "./data/model/biLstm_model_30"
    result_path = './data/result/biLstm_result.txt'
    train(train_corpus_path, labels_path, biLstm_model_path, result_path)


if __name__ == "__main__":
    main()
