import tensorflow as tf
import constants as cnst


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2D(x, W, ksize, strides):
    return tf.nn.conv2d(
        x, W, ksize=ksize, strides=strides, padding='SAME'
    )


class NeuralNet(object):
    def __init__(self, sess):
        self.assignable_params = []
        self.build_tf_network()

    def build_tf_network(self):
        """ Build the actual TF network """
        self.x = tf.placeholder("float", shape=[
            None, cnst.RESIZED_IMAGE_H,
            cnst.RESIZED_IMAGE_W, cnst.NUM_FRAMES_PASSED
            # batch size * 84 * 84 * 3
        ])

        self.y_target = tf.placeholder("float", cnst.ACTION_SPACE_SIZE)

        self.W_conv1 = weight_variable([
            8, 8, cnst.NUM_FRAMES_PASSED, 32
        ])
        self.b_conv1 = bias_variable([32])
        stride_1 = [2, 2]

        h_conv1 = tf.nn.relu(conv2D(self.x, self.W_conv1, stride_1) + self.b_conv1)

        self.W_conv2 = weight_variable([4, 4, 32, 64])
        self.b_conv2 = bias_variable([64])
        stride_2 = [2, 2]

        h_conv2 = tf.nn.relu(conv2D(h_conv1, self.W_conv2, stride_2) + self.b_conv2)

        self.W_conv3 = weight_variable([3, 3, 64, 64])
        self.b_conv3 = bias_variable([64])
        stride_3 = [1, 1]

        h_conv3 = tf.nn.relu(conv2D(h_conv2, self.W_conv3, stride_3) + self.b_conv3)

        out_size_after_conv = cnst.RESIZED_IMAGE_H / 4 * cnst.RESIZED_IMAGE_W / 4 * 64
        h_conv3_flat = tf.reshape(h_conv3, [-1, out_size_after_conv])

        self.W_fc_1 = weight_variable([out_size_after_conv, 512])
        self.b_fc_1 = bias_variable([512])

        h_fc_1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc_1) + self.b_fc_1)

        self.W_fc_2 = weight_variable([512, cnst.ACTION_SPACE_SIZE])
        self.b_fc_2 = bias_variable([cnst.ACTION_SPACE_SIZE])

        self.y_our = tf.nn.softmax(tf.matmul(h_fc_1, self.W_fc_2) + self.b_fc_2)
        self.choice = tf.argmax(self.y_our)
        # TODO(mike): Some kind of argmax is definitely needed here

        self.loss = tf.reduce_sum(tf.square(self.y_our - self.y_target))

        # TODO(mike): Check these rates, dude
        self.train_step = tf.train.RMSPropOptimizer(0.1, 0.1).minimize(loss)

        self.net_params = [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_conv3, self.b_conv3,
            self.W_fc_1, self.b_fc_1,
            self.W_fc_2, self.b_fc_2
        ]

    def train(self, session, data):
        self.train_step.run(
            feed_dict={
                self.x: data[0],
                self.y_target: data[1]
            },
            session=session
        )

    def choose_action(self, session, data):
        action = self.choice.run(
            feed_dict={
                self.x: data
            },
            session=session
        )
        return action

    def copy_params(self, session, other):
        """
        :param session: Tensorflow session to
        :param other:
        :return:
        """
        for my_param, other_param in zip(self.net_params,
                                         other.net_params):

            session.run(my_param.assign(other_param))
