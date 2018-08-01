import tensorflow as tf

m1 = tf.placeholder(tf.float32)

m2 = tf.placeholder(tf.float32)

m3 = [2, 3, 1]

p = m1-m2-m3

with tf.Session() as sess:
    result = sess.run(p, feed_dict={m1: [3, 2, 1], m2: [2, 3, 4]})

print(result)
