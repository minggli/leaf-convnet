import tensorflow as tf
import pandas as pd
import numpy as np
from utils import delete_folders, extract, pic_resize, batch_iter, move_classified, generate_training_set
import warnings
import os
import pandas as pd
import re

warnings.filterwarnings('ignore')

# params

image_path = 'leaf/images/'
model_path = 'models/'
pid_label, pid_name, mapping, data = extract('leaf/train.csv')
pic_ids = sorted([int(i.name.replace('.jpg', '')) for i in os.scandir(image_path) if i.is_file() and i.name.endswith('.jpg')])
input_shape = (8, 8)
images = dict()
for i in pic_ids:
    images[i] = pic_resize(image_path + str(i) + '.jpg', input_shape, pad=True)
test = pd.read_csv('leaf/test.csv', index_col=['id'])
m = input_shape[0] * input_shape[1]  # num of flat array
n = len(set(pid_name.values()))
d = 3

# transform
input_data = generate_training_set(test, pid_label=None, std=True)

# load image into tensor

sess = tf.Session()

# declare placeholders

x = tf.placeholder(dtype=tf.float32, shape=[None, d, m], name='feature')  # pixels as features
y_ = tf.placeholder(dtype=tf.float32, shape=[None, n], name='label')  # 99 classes in 1D tensor

# declare variables

W = tf.Variable(tf.zeros([d, m, n]))
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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# First Convolution Layer
W_conv1 = weight_variable([5, 5, d, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1, input_shape[0], input_shape[1], d])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer
W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([2 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 2 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, n])
b_fc2 = bias_variable([n])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=.9, beta2=.999).minimize(cross_entropy)
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


delete_folders()

order = list(test.index)
test = list()
for pid in order:
    test.append(input_data[pid])

test_data = np.array(test)

ans = sess.run(tf.nn.softmax(y_conv), feed_dict={x: test_data, keep_prob: 1})

move_classified(test_order=order, pid_name=pid_name, ans=ans, mapping=mapping)

data = pd.DataFrame(data=ans, columns=mapping.values(), dtype=np.float32, index=order)
data.sort_index(ascending=True, inplace=True)
data.index.rename('id', inplace=True)
data.reset_index(drop=False, inplace=True)
data.to_csv('submission.csv', encoding='utf-8', header=True, index=False)

