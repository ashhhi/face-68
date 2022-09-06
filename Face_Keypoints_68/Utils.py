import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

x_train_savepath = './../Data/300W/x_train_save.npy'
y_train_savepath = './../Data/300W/y_train_save.npy'
x_test_savepath = './../Data/300W/x_test_save.npy'
y_test_savepath = './../Data/300W/y_test_save.npy'

def DataLoader():
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
            y_test_savepath) and os.path.exists(y_test_savepath):
        print("---------------------加载数据---------------------")
        x_train = np.load(x_train_savepath)
        y_train = np.load(y_train_savepath)
        x_test = np.load(x_test_savepath)
        y_test = np.load(y_test_savepath)

    else:
        print("---------------------还未制作数据集---------------------")

    return x_train,y_train,x_test,y_test

def Visual(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1,2,1)
    plt.plot(acc,label='Training Accuracy')
    plt.plot(val_acc,label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1,2,2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')

    plt.legend()
    plt.show()

def Save_history(path,history):
    df = pd.DataFrame(columns=['step', 'train Loss', 'training accuracy','val loss','val accuracy'])  # 列名
    df.to_csv(path, index=False)  # 路径可以根据需要更改
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    for i in range(len(loss)):
        L = [i,loss[i],accuracy[i],val_loss[i],val_accuracy[i]]
        data = pd.DataFrame([L])
        data.to_csv(path,mode='a',header=False,index=False)  #mode为a是追加


