import tensorflow as tf
import numpy as np

# initialize number of nodes for every layer
l0_in = 0
l0_out = 0

l1_in = 0
l1_out = 0

l2_in = 0
l2_out = 0

l3_in = 0
l3_out = 0


def network_model(data):
    X, Y = data  # divide into data and metadata

    # Layer 1 computes time dependent jerks
    l0_input = tf.concat([X, Y], axis=0)
    l0 = {
        'weights': tf.Variable(tf.random_uniform([l0_in, l0_out])),
        'biases': tf.Variable(tf.random_uniform([l0_out]))
    }
    l0_output = tf.add(tf.matmul(l0_input, l0['weights']), l0['biases'])

    # target prediction using metadata
    l1_input = X
    l1 = {
        'weights': tf.Variable(tf.random_uniform([l1_in, l1_out])),
        'biases': tf.Variable(tf.random_uniform([l1_out]))
    }
    l1_output = tf.add(tf.matmul(l1_input, l1['weights']), l1['biases'])

    # target prediction using data and time dependent jerks
    l2_input = tf.concat([Y, l0_output], axis=0)
    l2 = {
        'weights': tf.Variable(tf.random_uniform([l2_in, l2_out])),
        'biases': tf.Variable(tf.random_uniform([l2_out]))
    }
    l2_output = tf.add(tf.matmul(l2_input, l2['weights']), l2['biases'])

    # to exploit the depency between the two predictions
    l3_input = tf.concat([l1_output, l2_output], axis=0)
    l3 = {
        'weights': tf.Variable(tf.random_uniform([l3_in, l3_out])),
        'biases': tf.Variable(tf.random_uniform([l3_out]))
    }
    l3_output = tf.add(tf.matmul(l3_input, l3['weights']), l3['biases'])

    # return the final prediction
    output = l3_output
    return output
