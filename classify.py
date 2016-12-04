from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import pandas as pd
import numpy as np
from utils import delete_folders, extract, pic_resize, batch_iter, move_classified
import warnings
import os
import re
warnings.filterwarnings('ignore')

# params

dir_path = 'leaf/images/'
model_path = 'models/'
pid_label, pid_name, mapping = extract('leaf/train.csv')
pic_names = [i.name for i in os.scandir(dir_path) if i.is_file() and i.name.endswith('.jpg')]
input_shape = (384, 384)
m = input_shape[0] * input_shape[1]  # num of flat array
n = len(set(pid_name.values()))

# load image into tensor

sess = tf.Session()

# declare placeholders

x = tf.placeholder(dtype=tf.float32, shape=[None, m], name='feature')  # pixels as features
y_ = tf.placeholder(dtype=tf.float32, shape=[None, n], name='label')  # 99 classes in 1D tensor

# declare variables

W = tf.Variable(tf.zeros([m, n]))
b = tf.Variable(tf.zeros([n]))

y = tf.matmul(x, W) + b


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


# First Convolution Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, input_shape[0], input_shape[1], 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Third layer

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# Densely connected layer
W_fc1 = weight_variable([6 * 6 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, n])
b_fc2 = bias_variable([n])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)


# Saver obj

saver = tf.train.Saver()
model_names = [i.name for i in os.scandir(model_path) if i.is_file() and i.name.endswith('.meta')]
loop_num = re.findall("[0-9]", model_names.pop())[0]
new_saver = tf.train.import_meta_graph(model_path + "model_loop_{0}.ckpt.meta".format(loop_num))
new_saver.restore(save_path=tf.train.latest_checkpoint(model_path), sess=sess)

leaf_images = dict()  # temp dictionary of re-sized leaf images
test = list()  # array of image and label in 1D array
test_order = list()
train = list()
train_order = list()
total = list()
total_order = list()
delete_folders()

for filename in pic_names:

    pid = int(filename.split('.')[0])
    leaf_images[pid] = pic_resize(dir_path + filename, size=input_shape, pad=True)

    if pid in pid_label.keys():
        directory = dir_path + 'test'
        train.append(np.array(leaf_images[pid]).flatten())
        train_order.append(filename)
    else:
        directory = dir_path + 'test'
        test.append(np.array(leaf_images[pid]).flatten())
        test_order.append(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    total.append(np.array(leaf_images[pid]).flatten())
    total_order.append(filename)
    leaf_images[pid].save(directory + '/' + filename)

test = np.array(test)
train = np.array(train)
total = np.array(total)

ans = sess.run(y_conv, feed_dict={x: test, keep_prob: 1})
move_classified(test_order, ans, mapping)
data = pd.DataFrame(data=ans, columns=mapping.values(), dtype=np.float32, index=[int(i.split('.')[0]) for i in test_order])
data.sort_index(ascending=True, inplace=True)

data.to_csv('submission1.csv', encoding='utf-8', header=True, index=True)
