import numpy as np
import tensorflow as tf

import constants as cnst


def weight_variable(shape):
    # TODO(mike): Write docstring
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # TODO(mike): Write docstring
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2D(x, W, stride):
    # TODO(mike): Write docstring
    return tf.nn.conv2d(
        x, W, strides=[1, stride, stride, 1], padding='VALID'
    )


class NeuralNet(object):
    # TODO(mike): Write docstring
    def __init__(self):
        # TODO(mike): Write docstring
        self.session = tf.Session()
        self.initialize_learning_network()
        self.initialize_target_network()

        self.define_copying_operation()
        self.define_choose_action()
        self.define_loss_operations()

        self.session.run(tf.initialize_all_variables())

        self.reset_target_net()

        self.saver = tf.train.Saver()

    def save_net(self, timestep):
        self.saver.save(self.session, 'dqn/saved-nets', global_step=timestep)

    def initialize_learning_network(self):
        # TODO(mike): Write docstring
        (
            self.phi_in, self.Q,
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_conv3, self.b_conv3,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2
        ) = self.define_Q_network()

    def initialize_target_network(self):
        # TODO(mike): Write docstring
        (
            self.phi_inT, self.QT,
            self.W_conv1T, self.b_conv1T,
            self.W_conv2T, self.b_conv2T,
            self.W_conv3T, self.b_conv3T,
            self.W_fc1T, self.b_fc1T,
            self.W_fc2T, self.b_fc2T
        ) = self.define_Q_network()

    def define_copying_operation(self):
        # TODO(mike): Write docstring
        self.copy_target_net_operation = [
            self.W_conv1T.assign(self.W_conv1),
            self.b_conv1T.assign(self.b_conv1),
            self.W_conv2T.assign(self.W_conv2),
            self.b_conv2T.assign(self.b_conv2),
            self.W_conv3T.assign(self.W_conv3),
            self.b_conv3T.assign(self.b_conv3),
            self.W_fc1T.assign(self.W_fc1),
            self.b_fc1T.assign(self.b_fc1),
            self.W_fc2T.assign(self.W_fc2),
            self.b_fc2T.assign(self.b_fc2)
        ]

    @staticmethod
    def define_Q_network():
        """ Build the actual TF network """
        phi_in = tf.placeholder("float", shape=[
            None, cnst.RESIZED_IMAGE_H,
            cnst.RESIZED_IMAGE_W, cnst.NUM_FRAMES_PASSED
            # batch size * 84 * 84 * 3
        ])

        W_conv1 = weight_variable([8, 8, cnst.NUM_FRAMES_PASSED, 32])
        b_conv1 = bias_variable([32])

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])

        out_size_after_conv = 3136
        W_fc_1 = weight_variable([out_size_after_conv, 512])
        b_fc_1 = bias_variable([512])

        W_fc_2 = weight_variable([512, cnst.ACTION_SPACE_SIZE])
        b_fc_2 = bias_variable([cnst.ACTION_SPACE_SIZE])

        stride_1 = 4
        stride_2 = 2
        stride_3 = 1

        h_conv1 = tf.nn.relu(conv2D(phi_in, W_conv1, stride_1) + b_conv1)
        h_conv2 = tf.nn.relu(conv2D(h_conv1, W_conv2, stride_2) + b_conv2)
        h_conv3 = tf.nn.relu(conv2D(h_conv2, W_conv3, stride_3) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, out_size_after_conv])
        h_fc_1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc_1) + b_fc_1)
        Q = tf.nn.softmax(tf.matmul(h_fc_1, W_fc_2) + b_fc_2)

        return (
            phi_in, Q,
            W_conv1, b_conv1,
            W_conv2, b_conv2,
            W_conv3, b_conv3,
            W_fc_1, b_fc_1,
            W_fc_2, b_fc_2
        )

    def define_choose_action(self):
        # TODO(mike): Write docstring
        self._choose_action = self.Q

    def define_loss_operations(self):
        # TODO(mike): Write docstring
        self.y_target = tf.placeholder(
            "float",
            shape=[None],
            name="y_target"
        )
        self.action = tf.placeholder(
            "float",
            shape=[None, cnst.ACTION_SPACE_SIZE],
            name="action"
        )

        executed_action = tf.reduce_sum(
            tf.mul(self.Q, self.action),
            reduction_indices=1
        )
        loss = tf.reduce_mean(tf.square(self.y_target - executed_action))

        self.train_step = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(loss)
        # TODO(mike): move params to constants

    def train(self, data):
        # TODO(mike): Write docstring
        (
            train_phis,
            train_actions,
            train_rewards,
            train_next_phis,
            train_terminals
        ) = data
        y_target = []
        qval_train = self.q_target_for_state(train_next_phis)

        for i in xrange(cnst.TRAINING_BATCH_SIZE):
            if train_terminals[i]:
                y_target.append(train_rewards[i])
            else:
                y_target.append(
                    train_rewards[i] + cnst.DISCOUNT_FACTOR * np.max(qval_train[i])
                )

        self.train_step.run(
            feed_dict={
                self.phi_in: train_phis,   # shape is (32, 84, 84, 3)
                self.y_target: y_target,   # this is list of 32 values
                self.action: train_actions # shape is (32, 6)
            },
            session=self.session
        )

    def q_target_for_state(self, phi):
        # TODO(mike): Write docstring
        return self.QT.eval(
            feed_dict={
                self.phi_inT: phi
            },
            session=self.session
        )

    def choose_action(self, one_point_of_data):
        # TODO(mike): Write docstring
        resu = self._choose_action.eval(
            feed_dict={
                self.phi_in: [one_point_of_data]
            },
            session=self.session
        )
        return np.argmax(resu[0])

    def reset_target_net(self):
        # TODO(mike): Write docstring
        self.session.run(self.copy_target_net_operation)
