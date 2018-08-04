# Skip-Gram
import tensorflow as tf
import tensorflow.contrib as tfc


def word2vec(corpus, embedding_size, window):
    vocabulary = []

    for sentence in corpus:
        sentence = sentence.split(' ')
        vocabulary.append(sentence)

    vocabulary = list(set(vocabulary))
    vocab_size = len(vocabulary)

    input_vec = tf.placeholder(tf.float32, [None, vocab_size])
    output_vec = tf.placeholder(tf.float32, [1, vocab_size])

    embeddings = tf.Variable(
        tf.random_normal([vocab_size, embedding_size])
    )
    embed = tf.nn.embedding_lookup(embeddings, input_vec)

    output_weights = tf.Variable(
        tf.random_normal([embedding_size, vocab_size]) / tf.sqrt(embedding_size)
    )
    output_bias = tf.Variable(tf.random_normal([vocab_size]))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(output_weights, output_bias, output_vec, embed, 5, vocab_size)
    )
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # 문장에서 배치 뽑아내서 학습
