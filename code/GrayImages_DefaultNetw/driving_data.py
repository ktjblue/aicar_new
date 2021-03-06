import scipy.misc

import random
import csv
#from mlxtend.preprocessing import one_hot
import numpy as np
import config as cfg
from skimage.transform import resize
import cv2
from PIL import Image
import math
from datetime import datetime

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.csv
increased_path = '../dataset/20200512/3-increased'
with open(increased_path + '/data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print(row[0], row[1])
        xs.append(increased_path + '/' + row[0])
        ys.append(int(row[1]))
        
increased_path = '../dataset/20200513/3-increased'
with open(increased_path + '/data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print(row[0], row[1])
        xs.append(increased_path + '/' + row[0])
        ys.append(int(row[1]))        

increased_path = '../dataset/training-data-shortcut/3-increased'
with open(increased_path + '/data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print(row[0], row[1])
        xs.append(increased_path + '/' + row[0])
        ys.append(int(row[1]))
        
increased_path = '../dataset/training-data-stop/3-increased'
with open(increased_path + '/data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print(row[0], row[1])
        xs.append(increased_path + '/' + row[0])
        ys.append(int(row[1]))        


###ys = one_hot(ys, num_labels=4, dtype='int')

#print(np.reshape(ys, -1))


#get number of images
num_images = len(xs)
num_train_images = math.floor(num_images * 0.8)
num_val_images = num_images - num_train_images

#shuffle list of images
c = list(zip(xs, ys))
random.seed(datetime.now())
random.shuffle(c)
random.seed(random.random())
random.shuffle(c)
xs, ys = zip(*c)
"""
train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

"""
#train_xs = xs[:int(len(xs) * 1)]
#train_ys = ys[:int(len(xs) * 1)]
#val_xs = xs[-int(len(xs) * 1):]
#val_ys = ys[-int(len(xs) * 1):]

train_xs = xs[:num_train_images]
train_ys = ys[:num_train_images]
val_xs = xs[num_train_images:]
val_ys = ys[num_train_images:]

#num_train_images = len(train_xs)
#num_val_images = len(val_xs)

"""
def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(
                        scipy.misc.imread(
                            train_xs[(train_batch_pointer + i) % num_train_images])[cfg.modelheight:], [66, 200]
                        ) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out
"""

"""
def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(
                        scipy.misc.imread(
                            val_xs[(val_batch_pointer + i) % num_val_images])[cfg.modelheight:], [66, 200]
                        ) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
"""


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # x_out.append(cv2.resize(cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images])[cfg.modelheight:],
        ind = (train_batch_pointer + i) % num_train_images
        filepath = train_xs[ind]
        
        src = cv2.imread(filepath)[cfg.modelheight:]
        dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        # print('in driving_data.py...')
        # print(dst.shape)
        
        x_out.append(cv2.resize(dst, dsize=(200, 66)) / 255.0)
        y_out.append([train_ys[ind]])
        # 예시로 출력 해 보자
        # cv2.imshow("src : " + str(y_out[i]), x_out[i])
        # print('filepath : ' + filepath)
        # cv2.waitKey(0)

    train_batch_pointer += batch_size
    return x_out, y_out


def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        ind = (val_batch_pointer + i) % num_val_images
        filepath = val_xs[ind]
        
        src = cv2.imread(filepath)[cfg.modelheight:]
        dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        # print('in driving_data.py...')
        # print(dst.shape)
        
        x_out.append(cv2.resize(dst, dsize=(200, 66)) / 255.0)
        y_out.append([val_ys[ind]])
        # 예시로 출력 해 보자
        # cv2.imshow("src : " + str(y_out[i]), x_out[i])
        # print('filepath : ' + filepath)
        # cv2.waitKey(0)

    val_batch_pointer += batch_size
    return x_out, y_out


# THE END