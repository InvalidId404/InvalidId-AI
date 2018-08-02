# Skip-Gram
import tensorflow as tf


def word2vec(batches, embedding_size, vocab_size):
    embeddings = tf.Variable(
        tf.random_normal([vocab_size, embedding_size])
    )

    output_weights = tf.Variable(
        tf.random_normal([embedding_size, vocab_size])/tf.sqrt(embedding_size)
    )
    output_bias = tf.Variable(tf.random_normal([vocab_size]))

    inputs = tf.placeholder(tf.float32, [None])
    object_outputs = tf.placehoder(tf.float32, [None, 1])

    embed = tf.nn.embedding_lookup(embeddings, inputs)

    loss = tf.reduce_mean(
        tf.nn.nce_loss(output_weights, output_bias, object_outputs, embed, 5, vocab_size)
    )

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for batch in batches:
            sess.run(optimizer, feed_dict={inputs: batch[0], object_outputs: batch[1]})
