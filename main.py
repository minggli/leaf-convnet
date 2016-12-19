import tensorflow as tf
import numpy as np
from utilities import delete_folders, extract, pic_resize, batch_iter, transform, move_classified, generate_training_set
import pandas as pd
import functools
import operator
import sys
import os

# coding: utf-8

__author__ = 'Ming Li'

"""This app from Ming Li is for leaf classification on Kaggle."""

# parameters

MODEL_PATH = 'models/'
IMAGE_PATH = 'leaf/images/'
INPUT_PATH = 'leaf/'

num_ensemble = 5
train, label, data = extract(INPUT_PATH + 'train.csv', target='species')
input_shape = (8, 8)
m = functools.reduce(operator.mul, input_shape, 1)
n = len(set(label))

print(sys.argv[1:])

EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False
ENSEMBLE = num_ensemble if 'ENSEMBLE' in map(str.upper, sys.argv[1:]) else 1
IMAGE = True if 'IMAGE' in map(str.upper, sys.argv[1:]) else False

d = 3 if not IMAGE else 4

if IMAGE:
    images_lib = {k: pic_resize(IMAGE_PATH + str(k) + '.jpg', input_shape, pad=True) for k in range(1, 1585, 1)}
else:
    images_lib = None

train_data = transform(data=train, label=label, dim=d, pixels=images_lib, normalize=True)

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


def _train(train_iterator, valid_set, optimiser, metric, loss, drop_out=.5):

    print('\n\n\n\n starting neural network #{}... \n\n\n\n'. format(loop))

    valid_x, valid_y = zip(*valid_set)

    for batch in train_iterator:
        epoch = batch[0]
        i = batch[1]
        x_batch, y_batch = zip(*batch[2])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        optimiser.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: drop_out})

        if i % 5 == 0:
            valid_accuracy, loss_score = sess.run([metric, loss], feed_dict={x: valid_x, y_: valid_y, keep_prob: 1.0})
            print("loop {4}, epoch {2}, step {0}, validation accuracy {1:.4f}, loss {3:.4f}".format(i, valid_accuracy, epoch, loss_score, loop))


def evaluate(test, metric, valid_set):

    valid_x, valid_y = zip(*valid_set)

    new_saver = tf.train.import_meta_graph(MODEL_PATH + 'model_ensemble_loop_{0}.ckpt.meta'.format(loop))
    new_saver.restore(save_path=MODEL_PATH + 'model_ensemble_loop_{0}.ckpt'.format(loop), sess=sess)

    probability = sess.run(tf.nn.softmax(logits), feed_dict={x: test, keep_prob: 1.0})
    valid_accuracy, valid_probability = sess.run([metric, tf.nn.softmax(logits)], feed_dict={x: valid_x, y_: valid_y, keep_prob: 1.0})

    return probability, valid_accuracy, valid_probability


def submit(raw):

    move_classified(test_data=raw, train_data=data, columns=label.columns, index=test.index, path=IMAGE_PATH)

    df = pd.DataFrame(data=raw, columns=label.columns, dtype=np.float32, index=test.index)
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
        W_conv2 = weight_variable([3, 3, 32, 64])
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

        _, valid_set = \
            generate_training_set(data=train_data, test_size=0.90)

        _, valid_y = zip(*valid_set)

        probs = []
        val_accuracies = []
        val_probs = []

        _, _, test = extract(INPUT_PATH + 'test.csv')
        test_data = transform(data=test, label=None, dim=d, pixels=images_lib, normalize=True)

        for loop in range(ENSEMBLE):

            prob, val_accuracy, val_prob = evaluate(test=test_data, metric=accuracy, valid_set=valid_set)
            probs.append(prob)
            val_accuracies.append(val_accuracy)
            val_probs.append(val_prob)

            print('Network: {0}, Validation Accuracy: {1:.4f}'.format(loop, val_accuracy))

        ensemble_val_prob = np.mean(np.array([val_probs[i] for i in range(ENSEMBLE)]), axis=0)
        ensemble_val_accuracy = sum(ensemble_val_prob.argmax(axis=1) == np.array(valid_y).argmax(axis=1)) / len(valid_y)

        print('Ensemble Network of ({0}), Validation Accuracy: {1:.4f}'.format(loop + 1, ensemble_val_accuracy))

        ensemble_prob = np.mean(np.array([probs[i] for i in range(ENSEMBLE)]), axis=0)
        submit(raw=ensemble_prob)

    else:

        for loop in range(ENSEMBLE):

            train_set, valid_set = \
                generate_training_set(data=train_data, test_size=0.20)

            batches = batch_iter(data=train_set, batch_size=200, num_epochs=3000, shuffle=True)

            with sess.as_default():
                sess.run(initializer)
                _train(train_iterator=batches, valid_set=valid_set, optimiser=train_step,
                       metric=accuracy, loss=loss, drop_out=.5)

            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)

            save_path = saver.save(sess, MODEL_PATH + "model_ensemble_loop_{0}.ckpt".format(loop))
            print("Model saved in file: {0}".format(save_path))
