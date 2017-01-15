from utilities import delete_folders, extract, pic_resize, batch_iter, transform, move_classified, generate_training_set
import pandas as pd
import functools
import operator
import sys
import os
import numpy as np
import tensorflow as tf

# coding: utf-8

__author__ = 'Ming Li'

"""This app from Ming Li is for leaf classification on Kaggle."""

# parameters

MODEL_PATH = 'models/'
IMAGE_PATH = 'leaf/images/'
INPUT_PATH = 'leaf/'

num_ensemble = 5
train, label, data = extract(INPUT_PATH + 'train.csv', target='species')

print(sys.argv[1:])

EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False
ENSEMBLE = num_ensemble if 'ENSEMBLE' in map(str.upper, sys.argv[1:]) else 1

input_shape = (1, 192)
d = 1

m = functools.reduce(operator.mul, input_shape, 1)
n = len(set(label))

train_data = transform(data=train, label=label, dim=d, input_shape=m, pixels=None, normalize=True)

# construct Deep Neural Network

default = {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 1e-3,
        'test_size': .20,
        'batch_size': 192,
        'num_epochs': 124,
        'drop_out': .3
    }

ensemble_hyperparams = {

    0: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 1e-4,
        'test_size': .20,
        'batch_size': 192,
        'num_epochs': 2000,
        'drop_out': .3
    },
    1: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 5e-5,
        'test_size': .20,
        'batch_size': 200,
        'num_epochs': 5000,
        'drop_out': .3
    },
    2: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 5e-5,
        'test_size': .20,
        'batch_size': 200,
        'num_epochs': 5000,
        'drop_out': .3
    },
    3: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 5e-5,
        'test_size': .20,
        'batch_size': 200,
        'num_epochs': 5000,
        'drop_out': .3
    },
    4: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 5e-5,
        'test_size': .20,
        'batch_size': 200,
        'num_epochs': 5000,
        'drop_out': .3
    }
}


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def graph(hyperparams):

    global logits
    global keep_prob
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

    x = tf.placeholder(dtype=tf.float32, shape=[None, m], name='feature')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, n], name='label')

    with tf.name_scope('hidden_layer_1'):
        W_hidden1 = weight_variable(hyperparams['hidden_layer_1'][0])
        b_hidden1 = bias_variable(hyperparams['hidden_layer_1'][1])

        h_hidden1 = tf.nn.relu(tf.matmul(x, W_hidden1) + b_hidden1)

    with tf.name_scope('drop_out_1'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_hidden1, keep_prob)

    with tf.name_scope('hidden_layer_2'):
        W_hidden2 = weight_variable(hyperparams['hidden_layer_2'][0])
        b_hidden2 = bias_variable(hyperparams['hidden_layer_2'][1])

        h_hidden2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_hidden2) + b_hidden2)

    with tf.name_scope('drop_out_2'):
        h_fc2_drop = tf.nn.dropout(h_hidden2, keep_prob)

    with tf.name_scope('read_out'):
        W_fc = weight_variable(hyperparams['read_out'][0])
        b_fc = bias_variable(hyperparams['read_out'][1])

        logits = tf.matmul(h_fc2_drop, W_fc) + b_fc

    # train
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
    loss = tf.reduce_mean(cross_entropy)
    # train_step = tf.train.AdamOptimizer(learning_rate=hyperparams['alpha'], beta1=.9, beta2=.99).minimize(loss)
    train_step = tf.train.RMSPropOptimizer(learning_rate=hyperparams['alpha']).minimize(loss)

    # eval
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # miscellaneous
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()


def optimise(train_iterator, valid_set, optimiser, metric, loss, drop_out=.3):

    print('\n\n\n\nstarting neural network #{}... \n'. format(loop))

    for i in sorted(default):
        print('{0}:{1}'.format(i, default[i]), end='\n', flush=False)
    print('\n', flush=True)

    valid_x, valid_y = zip(*valid_set)

    for batch in train_iterator:
        epoch = batch[0]
        i = batch[1]
        x_batch, y_batch = zip(*batch[2])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        optimiser.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: drop_out})

        if i % 5 == 0:
            valid_accuracy, loss_score = \
                sess.run([metric, loss], feed_dict={x: valid_x, y_: valid_y, keep_prob: 1.0})
            print("loop {4}, epoch {2}, step {0}, validation accuracy {1:.4f}, loss {3:.4f}".
                  format(i, valid_accuracy, epoch, loss_score, loop))


def evaluate(test, metric, valid_set):

    valid_x, valid_y = zip(*valid_set)

    new_saver = tf.train.import_meta_graph(MODEL_PATH + 'model_ensemble_loop_{0}.ckpt.meta'.format(loop))
    new_saver.restore(save_path=MODEL_PATH + 'model_ensemble_loop_{0}.ckpt'.format(loop), sess=sess)

    probability = sess.run(tf.nn.softmax(logits), feed_dict={x: test, keep_prob: 1.0})
    valid_accuracy, valid_probability = \
        sess.run([metric, tf.nn.softmax(logits)], feed_dict={x: valid_x, y_: valid_y, keep_prob: 1.0})

    return probability, valid_accuracy, valid_probability


def submit(raw):

    delete_folders()

    move_classified(test_data=raw, train_data=data, columns=label.columns, index=test.index, path=IMAGE_PATH)

    df = pd.DataFrame(data=raw, columns=label.columns, index=test.index)
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
        test_data = transform(data=test, label=None, dim=d, input_shape=m, pixels=None, normalize=True)

        for loop in range(ENSEMBLE):

            # establish a new Graph for a fresh session in Tensorflow

            g = tf.Graph()

            with g.as_default():

                graph(default)

                prob, val_accuracy, val_prob = evaluate(test=test_data, metric=accuracy, valid_set=valid_set)
                probs.append(prob)
                val_accuracies.append(val_accuracy)
                val_probs.append(val_prob)

            print('Network: {0}, Validation Accuracy: {1:.4f}'.format(loop, val_accuracy))

        ensemble_val_prob = np.mean(val_probs, axis=0)
        ensemble_val_accuracy = sum(ensemble_val_prob.argmax(axis=1) == np.array(valid_y).argmax(axis=1)) / len(valid_y)

        print('Ensemble Network of ({0}), Validation Accuracy: {1:.4f}'.format(loop + 1, ensemble_val_accuracy))

        ensemble_prob = np.mean(probs, axis=0)
        submit(raw=ensemble_prob)

    else:

        for loop in range(ENSEMBLE):

            train_set, valid_set = \
                generate_training_set(data=train_data, test_size=default['test_size'])

            batches = batch_iter(data=train_set, batch_size=default['batch_size'],
                                 num_epochs=default['num_epochs'], shuffle=True)

            g = tf.Graph()

            with g.as_default():
                graph(default)

                with sess.as_default():
                    sess.run(initializer)
                    optimise(train_iterator=batches, valid_set=valid_set, optimiser=train_step,
                           metric=accuracy, loss=loss, drop_out=default['drop_out'])

            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)

            save_path = saver.save(sess, MODEL_PATH + "model_ensemble_loop_{0}.ckpt".format(loop))
            print("Model saved in file: {0}".format(save_path))

