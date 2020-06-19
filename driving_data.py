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

train_xs = xs[:num_train_images]
train_ys = ys[:num_train_images]
val_xs = xs[num_train_images:]
val_ys = ys[num_train_images:]


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        ind = (train_batch_pointer + i) % num_train_images
        filepath = train_xs[ind]
        
        src = cv2.imread(filepath)[cfg.modelheight:]
        dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        x_out.append(cv2.resize(dst, dsize=(200, 66)) / 255.0)
        y_out.append([train_ys[ind]])

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
        
        x_out.append(cv2.resize(dst, dsize=(200, 66)) / 255.0)
        y_out.append([val_ys[ind]])


    val_batch_pointer += batch_size
    return x_out, y_out


# THE END