import tensorflow as tf
import numpy as np
import config as cfg
###import scipy

"""
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
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex
"""

color_channels = 1
x = tf.placeholder(tf.float32, shape=[None, 66, 200])

###y_ = tf.placeholder(tf.float32, shape=[None, 1])
y_ = tf.placeholder(tf.float32, shape=[None,  1])

image_vector_size = 66 * 200  # = 13,320
x_image = tf.reshape(x, [-1, image_vector_size])  # flat 하게 변환해줌

keep_prob = tf.placeholder(tf.float32)

WITH_BIAS = False

print('BIAS가 없는 ANN 구조가 더 좋은 결과를 보이는 것 같은데... 확인해 보자')

if WITH_BIAS == True:  # BIAS 값을 더한 ANN 구조
    # Layer 1 : input layer
    L1OutNeurons = 1024
    W1 = tf.Variable(tf.random_normal([image_vector_size, L1OutNeurons], mean = 0.0, stddev=0.01))
    b1 = tf.Variable(tf.random_normal([L1OutNeurons], mean = 0.0, stddev=0.01))
    L1 = tf.nn.relu(tf.matmul(x_image, W1) + b1)
    # 텐서플로우에 내장된 함수를 이용하여 dropout 을 적용합니다.
    # 함수에 적용할 레이어와 확률만 넣어주면 됩니다. 겁나 매직!!
    L1 = tf.nn.dropout(L1, keep_prob)

    # Layer 2 : hidden layer 1
    L2OutNeurons = 512
    W2 = tf.Variable(tf.random_normal([L1OutNeurons, L2OutNeurons], mean = 0.0, stddev=0.01))
    b2 = tf.Variable(tf.random_normal([L2OutNeurons], mean = 0.0, stddev=0.01))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob)

    # Layer 3 : hidden layer 2
    L3OutNeurons = 128
    W3 = tf.Variable(tf.random_normal([L2OutNeurons, L3OutNeurons], mean = 0.0, stddev=0.01))
    b3 = tf.Variable(tf.random_normal([L3OutNeurons], mean = 0.0, stddev=0.01))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob)

    # Output layer
    W4 = tf.Variable(tf.random_normal([L3OutNeurons, cfg.NUM_KEYS], mean = 0.0, stddev=0.01))
    b4 = tf.Variable(tf.random_normal([cfg.NUM_KEYS], mean = 0.0, stddev=0.01))
    y = tf.matmul(L3, W4) + b4
    
else:  # BIAS 값을 제외한 ANN 구조
    # Layer 1 : input layer
    L1OutNeurons = 1024
    W1 = tf.Variable(tf.random_normal([image_vector_size, L1OutNeurons], mean = 0.0, stddev=0.01))
    L1 = tf.nn.relu(tf.matmul(x_image, W1))
    # 텐서플로우에 내장된 함수를 이용하여 dropout 을 적용합니다.
    # 함수에 적용할 레이어와 확률만 넣어주면 됩니다. 겁나 매직!!
    L1 = tf.nn.dropout(L1, keep_prob)

    # Layer 2 : hidden layer 1
    L2OutNeurons = 512
    W2 = tf.Variable(tf.random_normal([L1OutNeurons, L2OutNeurons], mean = 0.0, stddev=0.01))
    L2 = tf.nn.relu(tf.matmul(L1, W2))
    L2 = tf.nn.dropout(L2, keep_prob)

    # Layer 3 : hidden layer 2
    L3OutNeurons = 128
    W3 = tf.Variable(tf.random_normal([L2OutNeurons, L3OutNeurons], mean = 0.0, stddev=0.01))
    L3 = tf.nn.relu(tf.matmul(L2, W3))
    L3 = tf.nn.dropout(L3, keep_prob)

    # Output layer
    W4 = tf.Variable(tf.random_normal([L3OutNeurons, cfg.NUM_KEYS], mean = 0.0, stddev=0.01))
    y = tf.matmul(L3, W4)

    
    
    
"""
#first convolutional layer
W_conv1 = weight_variable([5, 5, color_channels, 24])
b_conv1 = bias_variable([24])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)

#second convolutional layer
W_conv2 = weight_variable([5, 5, 24, 36])
b_conv2 = bias_variable([36])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

#third convolutional layer
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable([48])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)

#fourth convolutional layer
W_conv4 = weight_variable([3, 3, 48, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

#fifth convolutional layer
W_conv5 = weight_variable([3, 3, 64, 64])
b_conv5 = bias_variable([64])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)

#FCL 1
W_fc1 = weight_variable([1152, 1164])
b_fc1 = bias_variable([1164])

h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)


h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#FCL 2
W_fc2 = weight_variable([1164, 100])
b_fc2 = bias_variable([100])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#FCL 3
W_fc3 = weight_variable([100, 50])
b_fc3 = bias_variable([50])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

#FCL 3
W_fc4 = weight_variable([50, 10])
b_fc4 = bias_variable([10])

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

#Output
#W_fc5 = weight_variable([10, 4])
#b_fc5 = bias_variable([4])
W_fc5 = weight_variable([10, cfg.NUM_KEYS])
b_fc5 = bias_variable([cfg.NUM_KEYS])

#y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
###y = tf.mul(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
y = tf.matmul(h_fc4_drop, W_fc5) + b_fc5

#y = tf.nn.softmax(tf.matmul(h_fc4_drop, W_fc5) + b_fc5)


#print('y:',y.shape)
#print('y_:',y_.shape)
"""

print('model read')

# END