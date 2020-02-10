import tensorflow as tf

# 参考博客： https://suyuanliu.github.io/2018/12/03/TF-load-multipal-models/

class ImportGraph:
    def __init__(self, loc):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.xs = tf.placeholder(tf.float32, [None, 16, 152, 3], name='inputx')
            self.keep_prob = tf.placeholder(tf.float32, name='keepprob')
            self.output = self.model()  # model is a network function
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.saver.restore(self.sess, loc)

    def model(self):
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



