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

num_ensemble = 7
train, label, data = extract(INPUT_PATH + 'train.csv', target='species')
input_shape = (8, 8)
m = functools.reduce(operator.mul, input_shape, 1)
n = len(set(label))
print(sys.argv[1:])

EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False
ENSEMBLE = num_ensemble if 'ENSEMBLE' in map(str.upper, sys.argv[1:]) else 1
IMAGE = True if 'IMAGE' in map(str.upper, sys.argv[1:]) else False
IMAGEONLY = True if 'IMAGEONLY' in map(str.upper, sys.argv[1:]) else False

if IMAGEONLY:
    d = 1
elif IMAGE:
    d = 4
else:
    d = 3

images_lib = {k: pic_resize(IMAGE_PATH + str(k) + '.jpg', input_shape, pad=True) for k in range(1, 1585, 1)} \
    if IMAGE or IMAGEONLY else None

train_data = transform(data=train, label=label, dim=d, pixels=images_lib, normalize=True)

default = {
    'hidden_layer_1': [[5, 5, d, 32], [32]],
    'hidden_layer_2': [[5, 5, 32, 64], [64]],
    'dense_conn_1': [[2 * 2 * 64, 1024], [1024], [-1, 2 * 2 * 64]],
    'dense_conn_2': [[2048, 1024], [1024]],
    'read_out': [[1024, n], [n]]
}

ensemble_hyperparams = {

    0: {
        'hidden_layer_1': [[5, 5, d, 32], [32]],
        'hidden_layer_2': [[5, 5, 32, 64], [64]],
        'dense_conn_1': [[2 * 2 * 64, 1024], [1024], [-1, 2 * 2 * 64]],
        'dense_conn_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'test_size': .15,
        'batch_size': 200,
        'num_epochs': 3000,
        'drop_out': [.3, .25]
    },
    1: {
        'hidden_layer_1': [[5, 5, d, 64], [64]],
        'hidden_layer_2': [[5, 5, 64, 128], [128]],
        'dense_conn_1': [[2 * 2 * 128, 2048], [2048], [-1, 2 * 2 * 128]],
        'dense_conn_2': [[2048, 1024], [1024]],
        'read_out': [[1024, n], [n]],
        'test_size': .15,
        'batch_size': 300,
        'num_epochs': 3000,
        'drop_out': [.5, .5]
    },
    2: {
        'hidden_layer_1': [[5, 5, d, 32], [32]],
        'hidden_layer_2': [[5, 5, 32, 64], [64]],
        'dense_conn_1': [[2 * 2 * 64, 2048], [2048], [-1, 2 * 2 * 64]],
        'dense_conn_2': [[2048, 1024], [1024]],
        'read_out': [[1024, n], [n]],
        'test_size': .15,
        'batch_size': 200,
        'num_epochs': 3000,
        'drop_out': [.3, .25]
    },
    3: {
        'hidden_layer_1': [[5, 5, d, 64], [64]],
        'hidden_layer_2': [[5, 5, 64, 128], [128]],
        'dense_conn_1': [[2 * 2 * 128, 2048], [2048], [-1, 2 * 2 * 128]],
        'dense_conn_2': [[2048, 1024], [1024]],
        'read_out': [[1024, n], [n]],
        'test_size': .10,
        'batch_size': 250,
        'num_epochs': 3000,
        'drop_out': [.5, .5]
    },
    4: {
        'hidden_layer_1': [[5, 5, d, 32], [32]],
        'hidden_layer_2': [[5, 5, 32, 64], [64]],
        'dense_conn_1': [[2 * 2 * 64, 1024], [1024], [-1, 2 * 2 * 64]],
        'dense_conn_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'test_size': .10,
        'batch_size': 300,
        'num_epochs': 3000,
        'drop_out': [.3, .25]
    },
    5: {
        'hidden_layer_1': [[5, 5, d, 64], [64]],
        'hidden_layer_2': [[5, 5, 64, 128], [128]],
        'dense_conn_1': [[2 * 2 * 128, 1024], [1024], [-1, 2 * 2 * 128]],
        'dense_conn_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'test_size': .15,
        'batch_size': 300,
        'num_epochs': 3000,
        'drop_out': [.5, .5]
    },
    6: {
        'hidden_layer_1': [[5, 5, d, 64], [64]],
        'hidden_layer_2': [[5, 5, 64, 128], [128]],
        'dense_conn_1': [[2 * 2 * 128, 2048], [2048], [-1, 2 * 2 * 128]],
        'dense_conn_2': [[2048, 1024], [1024]],
        'read_out': [[1024, n], [n]],
        'test_size': .10,
        'batch_size': 300,
        'num_epochs': 3000,
        'drop_out': [.3, .25]
    }
}

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


def graph(hyperparams):

    global logits
    global keep_prob_1
    global keep_prob_2
    global accuracy
    global train_step
    global loss
    global sess
    global initializer
    global saver
    global x
    global y_

    sess = tf.Session()

    # declare placeholders

    x = tf.placeholder(dtype=tf.float32, shape=[None, d, m], name='feature')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, n], name='label')

    # reshaping input

    x_image = tf.reshape(x, [-1, input_shape[0], input_shape[1], d])

    with tf.name_scope('hidden_layer_1'):
        W_conv1 = weight_variable(hyperparams['hidden_layer_1'][0])
        b_conv1 = bias_variable(hyperparams['hidden_layer_1'][1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool(h_conv1)

    with tf.name_scope('hidden_layer_2'):
        W_conv2 = weight_variable(hyperparams['hidden_layer_2'][0])
        b_conv2 = bias_variable(hyperparams['hidden_layer_2'][1])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool(h_conv2)

    with tf.name_scope('dense_conn_1'):
        W_fc1 = weight_variable(hyperparams['dense_conn_1'][0])
        b_fc1 = bias_variable(hyperparams['dense_conn_1'][1])

        h_pool2_flat = tf.reshape(h_pool2, hyperparams['dense_conn_1'][2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('drop_out_1'):
        keep_prob_1 = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_1)

    with tf.name_scope('dense_conn_2'):
        W_fc2 = weight_variable(hyperparams['dense_conn_2'][0])
        b_fc2 = bias_variable(hyperparams['dense_conn_2'][1])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.name_scope('drop_out_2'):
        keep_prob_2 = tf.placeholder(tf.float32)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_2)

    with tf.name_scope('read_out'):
        W_fc3 = weight_variable(hyperparams['read_out'][0])
        b_fc3 = bias_variable(hyperparams['read_out'][1])

        logits = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    # train
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=.9, beta2=.99).minimize(loss)

    # eval
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # miscellaneous
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()


def optimise(train_iterator, valid_set, optimiser, metric, loss, drop_out=[.5, .5]):

    print('\n\n\n\n starting neural network #{}... \n'. format(loop))

    for i in ensemble_hyperparams[loop]:
        print('{0}:{1}'.format(i, ensemble_hyperparams[loop][i]))

    valid_x, valid_y = zip(*valid_set)

    for batch in train_iterator:
        epoch = batch[0]
        i = batch[1]
        x_batch, y_batch = zip(*batch[2])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        optimiser.run(feed_dict={x: x_batch, y_: y_batch, keep_prob_1: drop_out[0], keep_prob_2: drop_out[1]})

        if i % 5 == 0:
            valid_accuracy, loss_score = \
                sess.run([metric, loss], feed_dict={x: valid_x, y_: valid_y, keep_prob_1: 1.0, keep_prob_2: 1.0})
            print("loop {4}, epoch {2}, step {0}, validation accuracy {1:.4f}, loss {3:.4f}".
                  format(i, valid_accuracy, epoch, loss_score, loop))


def evaluate(test, metric, valid_set):

    valid_x, valid_y = zip(*valid_set)

    new_saver = tf.train.import_meta_graph(MODEL_PATH + 'model_ensemble_loop_{0}.ckpt.meta'.format(loop))
    new_saver.restore(save_path=MODEL_PATH + 'model_ensemble_loop_{0}.ckpt'.format(loop), sess=sess)

    probability = sess.run(tf.nn.softmax(logits), feed_dict={x: test, keep_prob_1: 1.0, keep_prob_2: 1.0})
    valid_accuracy, valid_probability = \
        sess.run([metric, tf.nn.softmax(logits)], feed_dict={x: valid_x, y_: valid_y, keep_prob_1: 1.0, keep_prob_2: 1.0})

    return probability, valid_accuracy, valid_probability


def submit(raw):

    delete_folders()

    move_classified(test_data=raw, train_data=data, columns=label.columns, index=test.index, path=IMAGE_PATH)

    df = pd.DataFrame(data=raw, columns=label.columns, dtype=np.float32, index=test.index)
    df.to_csv('submission.csv', encoding='utf-8', header=True, index=True)


if __name__ == '__main__':

    if EVAL:

        _, valid_set = \
            generate_training_set(data=train_data, test_size=0.80)

        _, valid_y = zip(*valid_set)

        probs = []
        val_accuracies = []
        val_probs = []

        _, _, test = extract(INPUT_PATH + 'test.csv')
        test_data = transform(data=test, label=None, dim=d, pixels=images_lib, normalize=True)

        for loop in range(ENSEMBLE):

            g = tf.Graph()

            with g.as_default():
                graph(ensemble_hyperparams[loop])

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
                generate_training_set(data=train_data, test_size=ensemble_hyperparams[loop]['test_size'])

            batches = batch_iter(data=train_set, batch_size=ensemble_hyperparams[loop]['batch_size'],
                                 num_epochs=ensemble_hyperparams[loop]['num_epochs'], shuffle=True)

            g = tf.Graph()

            with g.as_default():
                graph(ensemble_hyperparams[loop])

                with sess.as_default():
                    sess.run(initializer)
                    optimise(train_iterator=batches, valid_set=valid_set, optimiser=train_step,
                           metric=accuracy, loss=loss, drop_out=ensemble_hyperparams[loop]['drop_out'])

            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)

            save_path = saver.save(sess, MODEL_PATH + "model_ensemble_loop_{0}.ckpt".format(loop))
            print("Model saved in file: {0}".format(save_path))

