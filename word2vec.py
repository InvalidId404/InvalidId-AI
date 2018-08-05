# Skip-Gram
import tensorflow as tf
import tensorflow.contrib as tfc

def one_hot(value, length):
    return [1 if i == value else 0 for i in range(length)]


def word2vec(corpus, embedding_size, window):
    vocabulary = []

    for sentence in corpus:
        sentence = sentence.split(' ')

        if window > len(sentence)/2:
            return 0  # 나중에 예외처리

        vocabulary.append(sentence)

    vocabulary = {i: word for i, word in enumerate(list(set(vocabulary)))}
    vocabulary_byValue = {word: key for key, word in vocabulary.items()}
    vocab_size = len(vocabulary)

    # 문장에서 배치 뽑아내서 학습
    training_dataset = []
    for sentence in corpus:
        for w, word in enumerate(sentence):
            batch = sentence[w-window if w-window > 0 else 0 : w+window if w+window<len(sentence) else -1]
            batch.remove(word)
            batch = [[one_hot(vocabulary_byValue[word], vocab_size), one_hot(vocabulary_byValue[target], vocab_size)] for target in batch]
            training_dataset.append(batch)

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
    train_step = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for batch in training_dataset:
            x_batch = [b[0] for b in batch]
            y_batch = [b[1] for b in batch]
            sess.run(train_step, feed_dict={x_batch, y_batch})

            lookup_table = tfc.util.make_ndarray(embeddings)

    return lookup_table
