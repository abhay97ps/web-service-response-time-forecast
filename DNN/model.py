from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


class DNN:

    def fit(self, D_train, z_train, k):
        # shape samples * features
        X = D_train[:, :k]
        Y = D_train[:, k:]
        # k values decide the split between metadata and time stamped values
        self.k = k

        # initialize parameters
        self.l0_in = X.shape[1] + Y.shape[1]
        self.l0_out = Y.shape[1]
        self.l1_in = X.shape[1]
        self.l1_out = 1  # prediction using metadata
        self.l2_in = Y.shape[1] + self.l0_out
        self.l2_out = 1  # time series prediction
        self.l3_in = self.l1_out + self.l2_out
        self.l3_out = 1  # final prediction

        # splitting into train and validation set
        # validation set is kept as 1/5th of train set and is randomly selected
        self.D_train, self.D_validation, self.z_train, self.z_validation = train_test_split(
            D_train, z_train, test_size=0.2, random_state=1, shuffle=True)

        # define placeholder
        self.d = tf.placeholder('float', [None, D_train.shape[1]])
        self.z = tf.placeholder('float', [None, z_train.shape[1]])

        # train the model
        self.train()

    def network_model(self, data):
        # divide into data and metadata
        X = data[:, :self.k]
        Y = data[:, self.k:]
        # Layer 1 computes time dependent jerks
        l0_input = tf.concat([X, Y], axis=1)
        l0 = {
            'weights': tf.Variable(tf.random_uniform([self.l0_in, self.l0_out])),
            'biases': tf.Variable(tf.random_uniform([self.l0_out]))
        }
        l0_output = tf.add(tf.matmul(l0_input, l0['weights']), l0['biases'])

        # target prediction using metadata
        l1_input = X
        l1 = {
            'weights': tf.Variable(tf.random_uniform([self.l1_in, self.l1_out])),
            'biases': tf.Variable(tf.random_uniform([self.l1_out]))
        }
        l1_output = tf.add(tf.matmul(l1_input, l1['weights']), l1['biases'])

        # target prediction using data and time dependent jerks
        l2_input = tf.concat([Y, l0_output], axis=1)
        l2 = {
            'weights': tf.Variable(tf.random_uniform([self.l2_in, self.l2_out])),
            'biases': tf.Variable(tf.random_uniform([self.l2_out]))
        }
        l2_output = tf.add(tf.matmul(l2_input, l2['weights']), l2['biases'])

        # to exploit the depency between the two predictions
        l3_input = tf.concat([l1_output, l2_output], axis=1)
        l3 = {
            'weights': tf.Variable(tf.random_uniform([self.l3_in, self.l3_out])),
            'biases': tf.Variable(tf.random_uniform([self.l3_out]))
        }
        l3_output = tf.add(tf.matmul(l3_input, l3['weights']), l3['biases'])

        # return the final prediction
        output = l3_output
        return output

    def train(self):
        # get the prediction
        prediction = self.network_model(self.d)
        # cost function
        cost = tf.losses.mean_squared_error(
            labels=self.z, predictions=prediction)
        # regularization
        # yet to be added and make changes accordingly
        # train using adam optimizer
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        # initialize saver
        saver = tf.train.Saver()
        # session
        with tf.Session() as sess:
            # initializing the variables
            sess.run(tf.global_variables_initializer())
            # initial validation error
            error = sess.run(cost, feed_dict={
                             self.d: self.D_validation, self.z: self.z_validation})
            print('Initial Error: ', error)
            epochs = 5
            batch_size = 100
            num_batches = int(self.D_train.shape[0] / batch_size)
            for _ in range(epochs):
                # online training
                for i in range(0, num_batches, batch_size):
                    epoch_d = self.D_train[i:i+batch_size, :]
                    epoch_z = self.z_train[i:i+batch_size, :]
                    # training step
                    sess.run(optimizer, feed_dict={
                             self.d: epoch_d, self.z: epoch_z})
                error = sess.run(cost, feed_dict={
                                 self.d: self.D_validation, self.z: self.z_validation})
                print('Epoch error: ', error)
            # save the trained model
            saver.save(sess, './dnn_model.ckpt')

    def predict(self, D_test):
        prediction = self.network_model(self.d)
        saver = tf.train.Saver()
        # initialzing saver
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,"./dnn_model.ckpt")
            #saver = tf.train.import_meta_graph('dnn_model.meta')
            #saver.restore(sess, tf.train.latest_checkpoint('./'))
            #prediction = self.network_model(self.d)
            output = sess.run(prediction, feed_dict={self.d: D_test})
        return output
