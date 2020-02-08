import tensorflow as tf
import cv2
import os
import numpy as np

# redefine the network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# build the network
xs = tf.placeholder(tf.float32, [None, 14, 110, 3], name='inputx')
ys = tf.placeholder(tf.float32, [None, 3], name='inputy')
keep_prob = tf.placeholder(tf.float32, name='keepprob')
### conv1 ###
W_conv1 = weight_variable([1, 1, 3, 32])  # patch 3x3x3, 其中3是input image的n_channel；32是patch的个数=下一层的n_channel
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1)  # output size = 14x110x32
h_pool1 = max_pool_2x2(h_conv1)  # output size = 7x55x32
### func1 ###
W_fc1 = weight_variable([7*55*32, 1024])  # 全联接层的输入是把上一层的输出摊平
b_fc1 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool1, [-1, 7*55*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
### func2 ###
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    # print("weights=", sess.run(W_conv1))
    # print("bias=", sess.run(b_conv1))
    for root, dirs, files in os.walk("img/", topdown=False):
        for i in files:
            img = cv2.imread(root + "/" + i)
            x_img = img[np.newaxis, :]
            predict = sess.run(prediction, feed_dict={xs: x_img, keep_prob: 1})
            print("predict=", sess.run(tf.argmax(predict, 1)), "label=", i[-5])
