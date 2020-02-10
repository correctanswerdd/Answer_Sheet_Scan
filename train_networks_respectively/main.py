import tensorflow as tf
import cv2
import numpy as np
import pickle
from progressbar import *


def load_data(data_dir="data/"):
    with open(data_dir + 'X_train.pkl', 'rb') as f:
        X_train_ = pickle.load(f)
    with open(data_dir + 'y_train.pkl', 'rb') as f:
        y_train_ = pickle.load(f)
    with open(data_dir + 'X_test.pkl', 'rb') as f:
        X_test_ = pickle.load(f)
    with open(data_dir + 'y_test.pkl', 'rb') as f:
        y_test_ = pickle.load(f)
    with open(data_dir + 'X_val.pkl', 'rb') as f:
        X_val_ = pickle.load(f)
    with open(data_dir + 'y_val.pkl', 'rb') as f:
        y_val_ = pickle.load(f)
    return X_train_/255., y_train_, X_test_/255., y_test_, X_val_/255., y_val_


def compute_accuracy(validation_xs, validation_ys):
    global prediction
    vy_pred = sess.run(prediction, feed_dict={xs: validation_xs, ys: validation_ys, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(vy_pred, 1), tf.argmax(validation_ys, 1))  # argmax: 输出指定维度上最大数的坐标。1就是指定的维度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: validation_xs, ys: validation_ys, keep_prob: 1})
    return result


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
xs = tf.placeholder(tf.float32, [None, 16, 152, 3], name='inputx')
ys = tf.placeholder(tf.float32, [None, 4], name='inputy')
keep_prob = tf.placeholder(tf.float32, name='keepprob')
### conv1 ###
W_conv1 = weight_variable([2, 2, 3, 32])  # patch 2x2x3, 其中3是input image的n_channel；32是patch的个数=下一层的n_channel
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1)  # output size = 16, 152, 32
h_pool1 = max_pool_2x2(h_conv1)  # output size = 8x76x32
### func1 ###
W_fc1 = weight_variable([8*76*32, 1024])  # 全联接层的输入是把上一层的输出摊平
b_fc1 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool1, [-1, 8*76*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
### func2 ###
W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())


X_train, y_train, X_test, y_test, X_val, y_val = load_data()
### parameters
batch_size = 10
batch_num = X_train.shape[0] // batch_size
epochs = 10
best_val_loss = 100.0
time = 5
gl_alpha = 1
stop = False
###################

# training
t = 0
i = 0
while i < epochs and (not stop):
    i += 1
    for k in range(batch_num):
        batch_xs, batch_ys = X_train[k * batch_size: (k + 1) * batch_size, :], \
                             y_train[k * batch_size: (k + 1) * batch_size, :]
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if t == time:
            t = 0
            val_loss = sess.run(cross_entropy, feed_dict={xs: X_val, ys: y_val, keep_prob: 1})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                print("epoch %i, minibatch %i/%i, validation loss %f" % (i, k + 1, batch_num, val_loss))
                generalization_loss = (val_loss / best_val_loss) - 1
                if generalization_loss > gl_alpha:
                    print("    acc =", compute_accuracy(X_test, y_test))
                    stop = True
                    break
        else:
            t += 1


saver = tf.train.Saver()
save_path = saver.save(sess, "my_net/save_net.ckpt")
print("Save to path", save_path)
