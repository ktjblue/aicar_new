import tensorflow as tf
import numpy as np
###import scipy

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex


INPUT_IMAGE_HEIGHT = 66
INPUT_IMAGE_WIDTH = 200
INPUT_IMAGE_COLOR_CHANNEL = 3
x = tf.placeholder(tf.float32,
                   shape=[None, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_COLOR_CHANNEL])
###y_ = tf.placeholder(tf.float32, shape=[None, 1])
y_ = tf.placeholder(tf.float32, shape=[None,  1])

x_image = x

# first convolutional layer
CONV1_NUM_FEATURES = 24
W_conv1 = weight_variable([5, 5, INPUT_IMAGE_COLOR_CHANNEL, CONV1_NUM_FEATURES])
b_conv1 = bias_variable([CONV1_NUM_FEATURES])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)

# second convolutional layer
CONV2_NUM_FEATURES = 36
W_conv2 = weight_variable([5, 5, CONV1_NUM_FEATURES, CONV2_NUM_FEATURES])
b_conv2 = bias_variable([CONV2_NUM_FEATURES])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

#third convolutional layer
CONV3_NUM_FEATURES = 48
W_conv3 = weight_variable([5, 5, CONV2_NUM_FEATURES, CONV3_NUM_FEATURES])
b_conv3 = bias_variable([CONV3_NUM_FEATURES])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)

#fourth convolutional layer
CONV4_NUM_FEATURES = 64
W_conv4 = weight_variable([3, 3, CONV3_NUM_FEATURES, CONV4_NUM_FEATURES])
b_conv4 = bias_variable([CONV4_NUM_FEATURES])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

#fifth convolutional layer
CONV5_NUM_FEATURES = 64
W_conv5 = weight_variable([3, 3, CONV4_NUM_FEATURES, CONV5_NUM_FEATURES])
b_conv5 = bias_variable([CONV5_NUM_FEATURES])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)

#FCL 1
FC1_NUM_FEATUERS = 1164
W_fc1 = weight_variable([1152, FC1_NUM_FEATUERS])
b_fc1 = bias_variable([FC1_NUM_FEATUERS])

h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#FCL 2
FC2_NUM_FEATUERS = 100
W_fc2 = weight_variable([FC1_NUM_FEATUERS, FC2_NUM_FEATUERS])
b_fc2 = bias_variable([FC2_NUM_FEATUERS])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#FCL 3
FC3_NUM_FEATUERS = 50
W_fc3 = weight_variable([FC2_NUM_FEATUERS, FC3_NUM_FEATUERS])
b_fc3 = bias_variable([FC3_NUM_FEATUERS])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

#FCL 4
FC4_NUM_FEATUERS = 10
W_fc4 = weight_variable([50, FC4_NUM_FEATUERS])
b_fc4 = bias_variable([FC4_NUM_FEATUERS])

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

#Output
FC5_NUM_FEATUERS = 4
W_fc5 = weight_variable([FC4_NUM_FEATUERS, FC5_NUM_FEATUERS])
b_fc5 = bias_variable([FC5_NUM_FEATUERS])

#y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
###y = tf.mul(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
y = tf.matmul(h_fc4_drop, W_fc5) + b_fc5

#y = tf.nn.softmax(tf.matmul(h_fc4_drop, W_fc5) + b_fc5)


#print('y:',y.shape)
#print('y_:',y_.shape)
print('model read')

#C:\ProgramData\Anaconda3\lib;C:\ProgramData\Anaconda3\dll;C:\ProgramData\Anaconda3\scripts;