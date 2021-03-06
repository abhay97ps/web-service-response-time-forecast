from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


class DNN:

    def fit(self, D_train, z_train, k, D_test):
        # saving the test data
        self.D_test = D_test
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
        return self.train()

    def network_model(self, data):
        # divide into data and metadata
        X = data[:, :self.k]
        Y = data[:, self.k:]
        # weight initializer
        initializer = tf.contrib.layers.variance_scaling_initializer()
        # Layer 1 computes time dependent jerks
        l0_input = tf.concat([X, Y], axis=1)
        l0 = {
            'weights': tf.Variable(initializer([self.l0_in, self.l0_out])),
            'biases': tf.Variable(tf.random_uniform([self.l0_out]))
        }
        l0_output = tf.add(tf.matmul(l0_input, l0['weights']), l0['biases'])
        l0_output = tf.nn.dropout(tf.nn.relu(l0_output), keep_prob = 0.7)

        # target prediction using metadata
        l1_input = X
        l1 = {
            'weights': tf.Variable(initializer([self.l1_in, self.l1_out])),
            'biases': tf.Variable(tf.random_uniform([self.l1_out]))
        }
        l1_output = tf.add(tf.matmul(l1_input, l1['weights']), l1['biases'])
        l1_output = tf.nn.dropout(tf.nn.relu(l1_output), keep_prob = 0.7)

        # target prediction using data and time dependent jerks
        l2_input = tf.concat([Y, l0_output], axis=1)
        l2 = {
            'weights': tf.Variable(initializer([self.l2_in, self.l2_out])),
            'biases': tf.Variable(tf.random_uniform([self.l2_out]))
        }
        l2_output = tf.add(tf.matmul(l2_input, l2['weights']), l2['biases'])
        l2_output = tf.nn.dropout(tf.nn.relu(l2_output), keep_prob = 0.7)

        # to exploit the depency between the two predictions
        l3_input = tf.concat([l1_output, l2_output], axis=1)
        l3 = {
            'weights': tf.Variable(initializer([self.l3_in, self.l3_out])),
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
        # session
        with tf.Session() as sess:
            # initializing the variables
            sess.run(tf.global_variables_initializer())
            # initial validation error
            error = sess.run(cost, feed_dict={
                             self.d: self.D_validation, self.z: self.z_validation})
            print('Initial validation Error: ', error)
            # implementing early stopping for regularization
            epoch_steps = 5
            patience = 10
            cnt = 0
            batch_size = 100
            val_error = error
            total_epochs = 0
            num_batches = int(self.D_train.shape[0] / batch_size)
            while True:
                for _ in range(epoch_steps):
                    # online training
                    for i in range(0, num_batches, batch_size):
                        epoch_d = self.D_train[i:i+batch_size, :]
                        epoch_z = self.z_train[i:i+batch_size, :]
                        # training step
                        sess.run(optimizer, feed_dict={
                            self.d: epoch_d, self.z: epoch_z})
                error = sess.run(cost, feed_dict={
                                 self.d: self.D_validation, self.z: self.z_validation})
                if error < val_error:
                    val_error = error
                    cnt = 0
                    total_epochs += epoch_steps
                else:
                    cnt += 1
                    total_epochs += epoch_steps
                if cnt == patience or total_epochs == 1000:
                    break
                print('Validation error after ' +
                      str(total_epochs) + ' epochs is: ' + str(error))
            # returning the predictions for test file
            output = sess.run(prediction, feed_dict={self.d: self.D_test})
            return output
