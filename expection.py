import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def linear_regression(x_batch, y_batch, learning_rate, epoch):
    features = len(x_batch[0])

    input = tf.placeholder(tf.float32, [None, features])
    weight = tf.Variable(tf.random_normal([features, 1]))
    bias = tf.Variable(-1.)

    output = tf.matmul(input, weight) + bias
    y_batch = tf.constant(y_batch)

    loss = tf.reduce_mean(tf.square(y_batch-output))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(epoch):
            sess.run(train_step, feed_dict={input: x_batch})
        r = sess.run(weight)
        b = sess.run(bias)

    return r, b

def main():
    f = lambda x: 0.2*x+0.3

    dataset = []

    p = 5000

    for i in range(p):
        x = np.random.normal(0.0, 0.55)
        y = f(x) + np.random.normal(0.0, 0.03)
        dataset.append([x, y])

    x_data = [[data[0]] for data in dataset]
    y_data = [[data[1]] for data in dataset]

    print(linear_regression(x_data, y_data, learning_rate=0.5, epoch=10))


if __name__ == '__main__':
    main()
