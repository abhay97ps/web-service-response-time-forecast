import tensorflow as tf
import os
import random
import numpy as np
n_classes = 5
batch_size = 128
Input_path = "./../Input/Images_in/"

x = tf.placeholder('float', [None, 36])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

train_x = []
train_y = []
test_x = []
test_y = []
current_train_index = 0

def prepare_data():
	files = os.listdir(Input_path)
	print len(files)
	random.shuffle(files)
	#files = files[0:500]
	train_files = files[:int(0.75*len(files))]
	test_files = files[int(0.75*len(files)):]
	for item in train_files:
		one_hot = [0] * n_classes
		try:
			img_arr = np.load(Input_path + item)
			lab = item.split('.')[0].split('_')[1]
			train_x.append(img_arr.flatten())
			one_hot[int(lab)] = 1
			train_y.append(one_hot)
		except:
			print "$"
	for item in test_files:
		one_hot = [0] * n_classes
		try:
			img_arr = np.load(Input_path + item)
			lab = item.split('.')[0].split('_')[1]
			test_x.append(img_arr.flatten())
			one_hot[int(lab)] = 1
			test_y.append(one_hot)
		except:
			print "$"

def get_train_batch():
	global current_train_index
	if current_train_index + batch_size < len(train_x):
		current_train_index += batch_size
		return train_x[current_train_index-batch_size:current_train_index],train_y[current_train_index-batch_size:current_train_index]
	else:
		return None, None
def get_test_data():
	return [test_x,text_y]

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,1,32])),
               #'W_conv2':tf.Variable(tf.random_normal([3,3,32,64])),
               'W_fc':tf.Variable(tf.random_normal([3*3*32,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               #'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 6, 6, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    #conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    #conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv1,[-1, 3*3*32])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            global current_train_index
            current_train_index = 0
            epoch_loss = 0
            while True:
                epoch_x, epoch_y = get_train_batch()
		if epoch_x == None:
			break
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

prepare_data()
train_neural_network(x)
