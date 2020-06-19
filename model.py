import tensorflow as tf
import numpy as np
import config as cfg
import math
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
x_image = tf.reshape(x, [-1, 66, 200, color_channels])  # flat 하게 변환해줌

keep_prob = tf.placeholder(tf.float32)






WITH_BIAS = False  # CNN은 BIAS 필요 없음

if WITH_BIAS == True:
    assert False, 'CNN no needs BIAS'
else:
    # 각각의 변수와 레이어는 다음과 같은 형태로 구성됩니다.
    # W1 [3 3 1 32] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 32: 필터 갯수
    # L1 Conv shape=(?, height, width, 32)
    #    Pool     ->(?, 14, 14, 32)
    L1NumFilters = 32
    W1 = tf.Variable(tf.random_normal([3, 3, color_channels, L1NumFilters], mean=0.0, stddev=0.01))
    # tf.nn.conv2d 를 이용해 한칸씩 움직이는 컨볼루션 레이어를 쉽게 만들 수 있습니다.
    # padding='SAME' 은 커널 슬라이딩시 최외곽에서 한칸 밖으로 더 움직이는 옵션
    L1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    # Pooling 역시 tf.nn.max_pool 을 이용하여 쉽게 구성할 수 있습니다.
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # L1 = tf.nn.dropout(L1, keep_prob)


    # L2 Conv shape=(?, 14, 14, 64)
    #    Pool     ->(?, sizeHeight, sizeWidth, 64)
    # W2 의 [3, 3, 32, 64] 에서 32 는 L1 에서 출력된 W1 의 마지막 차원, 필터의 크기 입니다.
    L2NumFilters = 64
    W2 = tf.Variable(tf.random_normal([3, 3, L1NumFilters, L2NumFilters], mean=0.0, stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # L2 = tf.nn.dropout(L2, keep_prob)


    # FC 레이어: 입력값 7x7x64 -> 출력값 256
    # Full connect를 위해 직전의 Pool 사이즈인 (?, sizeHeight, sizeWidth, 64) 를 참고하여 차원을 줄여줍니다.
    #    Reshape  ->(?, 256)
    NumOutNeurons = 256
    #sizeHeight = 7  # 28 / 2 / 2
    sizeHeight = math.ceil(66 / 4)
    #sizeWidth = 7  # 28 / 2 / 2
    sizeWidth = round(200 / 4)
    W3 = tf.Variable(tf.random_normal([sizeHeight * sizeWidth * L2NumFilters, NumOutNeurons], mean=0.0, stddev=0.01))
    L3 = tf.reshape(L2, [-1, sizeHeight * sizeWidth * L2NumFilters])
    L3 = tf.matmul(L3, W3)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob)


    # 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
    W4 = tf.Variable(tf.random_normal([NumOutNeurons, cfg.NUM_KEYS], mean=0.0, stddev=0.01))
    y = tf.matmul(L3, W4)
    #y_temp = tf.matmul(L3, W4)
    #y = tf.nn.softmax(y_temp, axis=None, name=None)










"""
#
# ANN 에서 사용한 네트워크 구조
#

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
"""


"""
#
# 샘플 코드에서 사용한 네트워크 구조
#

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