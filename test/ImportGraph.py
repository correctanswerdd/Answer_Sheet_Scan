import tensorflow as tf


class ImportGraph:
    def __init__(self, loc, c:int):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.keep_prob = tf.placeholder(tf.float32, name='keepprob')
            if c == 3:
                self.xs = tf.placeholder(tf.float32, [None, 14, 110, 3], name='inputx')
                self.output = self.model3()  # model is a network function
            elif c == 4:
                self.xs = tf.placeholder(tf.float32, [None, 16, 152, 3], name='inputx')
                self.output = self.model4()  # model is a network function
            elif c == 7:
                self.xs = tf.placeholder(tf.float32, [None, 16, 232, 3], name='inputx')
                self.output = self.model7()  # model is a network function
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.saver.restore(self.sess, loc)

    def model4(self):
        ### conv1 ###
        W_conv1 = self.weight_variable(
            [2, 2, 3, 32])  # patch 2x2x3, 其中3是input image的n_channel；32是patch的个数=下一层的n_channel
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(self.xs, W_conv1) + b_conv1)  # output size = 16, 152, 32
        h_pool1 = self.max_pool_2x2(h_conv1)  # output size = 8x76x32
        ### func1 ###
        W_fc1 = self.weight_variable([8 * 76 * 32, 1024])  # 全联接层的输入是把上一层的输出摊平
        b_fc1 = self.bias_variable([1024])
        h_pool1_flat = tf.reshape(h_pool1, [-1, 8 * 76 * 32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        ### func2 ###
        W_fc2 = self.weight_variable([1024, 4])
        b_fc2 = self.bias_variable([4])
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        return prediction

    def model3(self):
        ### conv1 ###
        W_conv1 = self.weight_variable(
            [1, 1, 3, 32])  # patch 3x3x3, 其中3是input image的n_channel；32是patch的个数=下一层的n_channel
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(self.xs, W_conv1) + b_conv1)  # output size = 14x110x32
        h_pool1 = self.max_pool_2x2(h_conv1)  # output size = 7x55x32
        ### func1 ###
        W_fc1 = self.weight_variable([7 * 55 * 32, 1024])  # 全联接层的输入是把上一层的输出摊平
        b_fc1 = self.bias_variable([1024])
        h_pool1_flat = tf.reshape(h_pool1, [-1, 7 * 55 * 32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        ### func2 ###
        W_fc2 = self.weight_variable([1024, 3])
        b_fc2 = self.bias_variable([3])
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="predict")
        return prediction

    def model7(self):
        ### conv1 ###
        W_conv1 = self.weight_variable([2, 2, 3, 32])  # patch 2x2x3, 其中3是input image的n_channel；32是patch的个数=下一层的n_channel
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(self.xs, W_conv1) + b_conv1)  # output size = 16, 232, 32
        h_pool1 = self.max_pool_2x2(h_conv1)  # output size = 8x116x32
        ### conv2 ###
        W_conv2 = self.weight_variable([2, 2, 32, 64])  # patch 2x2x3, 其中3是input image的n_channel；32是patch的个数=下一层的n_channel
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)  # output size = 8, 116, 64
        h_pool2 = self.max_pool_2x2(h_conv2)  # output size = 4x58x64
        ### func1 ###
        W_fc1 = self.weight_variable([4 * 58 * 64, 1024])  # 全联接层的输入是把上一层的输出摊平
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 58 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        ### func2 ###
        W_fc2 = self.weight_variable([1024, 7])
        b_fc2 = self.bias_variable([7])
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        return prediction

    def predict(self, x):
        return self.sess.run(self.output, feed_dict={self.xs: x, self.keep_prob: 1})

    # redefine the network
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



