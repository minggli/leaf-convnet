import tensorflow as tf
import numpy as np
from sklearn import model_selection
from utilities import delete_folders, extract, pic_resize, batch_iter, generate_training_set, move_classified
import sys
import os

# coding: utf-8

__author__ = 'Ming Li'

"""This application forms a submission from Ming Li in regards to leaf classification on Kaggle."""

# parameters

try:
    EVAL = False if str(sys.argv[1]).upper() != 'EVAL' else True
except IndexError:
    EVAL = False

MODEL_PATH = 'models/'
IMAGE_PATH = 'leaf/images/'
INPUT_PATH = 'leaf/'

pid_label, pid_name, mapping, data = extract('leaf/train.csv')
pic_ids = sorted([int(i.name.replace('.jpg', '')) for i in os.scandir(IMAGE_PATH) if i.is_file() and i.name.endswith('.jpg')])
input_data = generate_training_set(data, pid_label=pid_label, std=True)
input_shape = (8, 8)
images = dict()

for i in pic_ids:
    images[i] = pic_resize(IMAGE_PATH + str(i) + '.jpg', input_shape, pad=True)
m = input_shape[0] * input_shape[1]  # num of flat array
n = len(set(pid_name.values()))
d = 3

# load image into tensor


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def _train(iterator, optimiser, metric, loss, drop_out=.5):

    print('\n\n\n\n starting cross validation... \n\n\n\n')

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    for batch in iterator:
        epoch = batch[0]
        i = batch[1]
        x_batch, y_batch = zip(*batch[2])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        optimiser.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: drop_out})

        if i % 5 == 0:
            train_accuracy, loss_score = sess.run([metric, loss], feed_dict={x: valid_x, y_: valid_y, keep_prob: 1.0})
            print("epoch {2}, step {0}, training accuracy {1:.4f}, loss {3:.4f}".format(i, train_accuracy, epoch, loss_score))

    save_path = saver.save(sess, MODEL_PATH + "model_epoch_{0}.ckpt".format(epoch))
    print("Model saved in file: {0}".format(save_path))


def _evaluate():

    import pandas as pd
    import re

    test = pd.read_csv(INPUT_PATH + 'test.csv', index_col='id')
    input_test = generate_training_set(test, pid_label=None, std=True)
    test_set = list()
    for i in test.index:
        test_set.append(input_test[i])

    model_names = [i.name for i in os.scandir(MODEL_PATH) if i.is_file() and i.name.endswith('.meta')]
    loop_num = re.findall("[0-9][0-9]*", model_names.pop())[0]
    new_saver = tf.train.import_meta_graph(MODEL_PATH + 'model_epoch_{0}.ckpt.meta'.format(loop_num))
    new_saver.restore(save_path=tf.train.latest_checkpoint(MODEL_PATH), sess=sess)

    probs = sess.run(tf.nn.softmax(logits), feed_dict={x: np.array(test_set), keep_prob: 1.0})

    move_classified(test_order=test.index, pid_name=pid_name, ans=probs, mapping=mapping)

    df = pd.DataFrame(data=probs, columns=mapping.values(), dtype=np.float32, index=test.index)
    df.to_csv('submission.csv', encoding='utf-8', header=True, index=True)


if __name__ == '__main__':

    sess = tf.Session()

    # declare placeholders

    x = tf.placeholder(dtype=tf.float32, shape=[None, d, m], name='feature')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, n], name='label')

    # declare weights and bias unit

    W = tf.Variable(tf.zeros([d, m, n]), name='weight')
    b = tf.Variable(tf.zeros([n]), name='bias')

    # reshaping input

    x_image = tf.reshape(x, [-1, input_shape[0], input_shape[1], d])

    with tf.name_scope('hidden_layer_1'):
        W_conv1 = weight_variable([5, 5, d, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool(h_conv1)

    with tf.name_scope('hidden_layer_2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool(h_conv2)

    with tf.name_scope('dense_conn_1'):
        W_fc1 = weight_variable([2 * 2 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 2 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('drop_out'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('read_out'):
        W_fc2 = weight_variable([1024, n])
        b_fc2 = bias_variable([n])

    # logits

    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # train
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=.9, beta2=.999).minimize(loss)

    # eval
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # miscellaneous
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()

if EVAL:

    _evaluate()

else:

    kf_iterator = model_selection.StratifiedKFold(n_splits=5, shuffle=False)  # Stratified

    train_x = list(pid_name.keys())  # leaf id
    train_y = list(pid_name.values())  # leaf species names

    for train_index, valid_index in kf_iterator.split(train_x, train_y):

        train = list()  # array of image and label in 1D array
        valid = list()  # array of image and label in 1D array

        train_id = [train_x[idx] for idx in train_index]
        valid_id = [train_x[idx] for idx in valid_index]

        for pid in train_x:

            if pid in train_id:
                train.append(input_data[pid])

            elif pid in valid_id:
                valid.append(input_data[pid])

        # create batches
        train = np.random.permutation(np.array(train))
        batches = batch_iter(data=train, batch_size=200, num_epochs=2000, shuffle=True)

        valid = np.array(valid)
        valid_x = np.array([i[0] for i in valid])
        valid_y = np.array([i[1] for i in valid])

        with sess.as_default():
            sess.run(initializer)
            _train(iterator=batches, optimiser=train_step, metric=accuracy, loss=loss, drop_out=.5)

        break
