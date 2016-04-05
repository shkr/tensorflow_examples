import tensorflow as tf
import numpy as np
import pylab as Plot
import time

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 400, 'Number of steps to run trainer.')
flags.DEFINE_float('perplexity', 30.0, 'The perplexity to set for each Gaussian Distribution around an input vector')
flags.DEFINE_float('output_dimensions', 2, 'The dimensions of the embedding vector')
flags.DEFINE_float('step_size', 0.01, 'The step size for Gradient Descent')
flags.DEFINE_float('binary_iterations', 50, 'The number of iteratiosn for binary search')
flags.DEFINE_float('learning_rate', 0.01, 'The dimensions of the embedding vector')
flags.DEFINE_float('momentum', 0.5, 'The dimensions of the embedding vector')
flags.DEFINE_float('optimizer_steps', 1000, 'The dimensions of the embedding vector')


def distance_matrix(graph, I):
    """

    :param I:
    :return:
    """
    with graph.as_default():
        I_mag_sqr = tf.reduce_sum(tf.mul(I, I), 1)
        I_sqr = tf.matmul(I , tf.transpose(I))

        # Distance Matrix where D(ij) is the Distance between x_i & x_j
        # D = ((-2X_sqr + X_mag_sqr)_T + X_mag_sqr)_T
        D = tf.transpose(tf.add(tf.transpose(tf.add(tf.scalar_mul(-2.0, I_sqr), I_mag_sqr)), I_mag_sqr))
    return D


def Y2Q(graph, Y):
    """
    :param Y:
    :return:
    """
    with graph.as_default():
        (no_rows, out_dimensions) = Y.get_shape().as_list()
        D = distance_matrix(graph, Y)
        ones = tf.ones(shape=[no_rows, no_rows], dtype=tf.float32)
        inverse_D = tf.truediv(ones, tf.add(ones, D))
        inverse_D_diag_zero = tf.pack([tf.concat(0, [inverse_D[i, 0:i], [0.0], inverse_D[i, i+1:no_rows]]) for i in range(no_rows)])
        inverse_sum = tf.tile(
            tf.reshape(tf.reduce_sum(inverse_D_diag_zero, 1), [no_rows, 1]),
            multiples=[1, no_rows]
        )
        Q = tf.mul(inverse_D, inverse_sum)
    return Q


def training(loss):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
    # Use the optimizer to apply the gradients that minimize the loss
    train_op = optimizer.minimize(loss)
    return train_op


def P2Y(P_):
    """

    :return:
    """
    (no_rows, no_cols) = P_.shape
    print("P matrix, given shape = ({}, {})".format(no_rows, no_cols))
    if no_rows != no_cols:
        raise Exception("P must be a square matrix, given shape = ({}, {})".format(no_rows, no_cols))

    graph = tf.Graph()
    with graph.as_default():

        given_P = tf.constant(value=P_, shape=[no_rows, no_cols], dtype=tf.float32)
        neg_one = tf.constant(-1.0, dtype=tf.float32)
        P_P_t = tf.add(given_P, tf.transpose(given_P))
        reduce_P = tf.reshape(tf.reduce_sum(P_P_t, reduction_indices=0), shape=[no_rows, 1])
        sum_P = tf.tile(reduce_P, multiples=[1, no_cols])
        inv_sum_P = tf.truediv(tf.ones(shape=[no_rows, no_cols], dtype=tf.float32), sum_P)

        # Initialize Flags
        max_iter = 1000
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = 500
        min_gain = 0.01
        # Early Exaggeration
        P = tf.scalar_mul(4, tf.mul(P_P_t, inv_sum_P))
        Y = tf.Variable(tf.truncated_normal(shape=[no_rows, FLAGS.output_dimensions], dtype=tf.float32))
        dY = tf.Variable(tf.truncated_normal(shape=[no_rows, FLAGS.output_dimensions], dtype=tf.float32))
        iY = tf.Variable(tf.truncated_normal(shape=[no_rows, FLAGS.output_dimensions], dtype=tf.float32))
        gains = tf.Variable(tf.truncated_normal(shape=[no_rows, FLAGS.output_dimensions], dtype=tf.float32))

        D = distance_matrix(graph, Y)
        ones = tf.ones(shape=[no_rows, no_rows], dtype=tf.float32)
        inverse_D = tf.truediv(ones, tf.add(ones, D))
        inverse_D_diag_zero = tf.pack([tf.concat(0, [inverse_D[i, 0:i], [0.0], inverse_D[i, i+1:no_rows]]) for i in range(no_rows)])
        inverse_sum = tf.tile(
            tf.reshape(tf.reduce_sum(inverse_D_diag_zero, 1), [no_rows, 1]),
            multiples=[1, no_rows]
        )
        Q = tf.mul(inverse_D, inverse_sum)
        
        for step in range(FLAGS.optimizer_steps):


            PQ = tf.sub(P, Q)


        kl_divergence = tf.reduce_sum(tf.scalar_mul(neg_one, tf.mul(P, tf.log(tf.div(P, Q)))), [0, 1])
        train_ops = training(kl_divergence)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        for step in range(FLAGS.optimizer_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_ops, kl_divergence], feed_dict={})
            duration = time.time() - start_time
                  # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

        Y_ = sess.run(Y)
        sess.close()
    return Y_


def D2P(D_):

    (no_rows, no_cols) = D_.shape
    print("D matrix, given shape = ({}, {})".format(no_rows, no_cols))
    P_ = np.zeros((no_rows, no_cols))
    beta_ = np.ones((no_rows, 1))
    if no_rows != no_cols:
        raise Exception("D must be a square matrix, given shape = ({}, {})".format(no_rows, no_cols))

    start_time = time.time()

    for i in range(no_rows):

        graph = tf.Graph()
        D_i = D_[i, np.concatenate((np.r_[0: i], np.r_[i+1: no_rows]))]

        with graph.as_default():
            d_i = tf.constant(value=D_i, shape=[1, no_cols - 1], dtype=tf.float32)
            inf = tf.constant(np.inf, dtype=tf.float32, shape=[])
            tol = 1e-5
            one = tf.constant(1.0, dtype=tf.float32, shape=[])
            neg_one = tf.constant(-1.0, dtype=tf.float32, shape=[])
            two = tf.constant(2.0, dtype=tf.float32, shape=[])
            logU = tf.constant(np.log(FLAGS.perplexity), dtype=tf.float32)
            # These 3 are 0-D Tensors
            beta_min = tf.Variable(initial_value=tf.neg(inf))
            beta_max = tf.Variable(initial_value=inf)
            beta = tf.Variable(initial_value=tf.constant(1.0, dtype=tf.float32, shape=[]))

            exp_i = tf.exp(tf.scalar_mul(beta, tf.scalar_mul(neg_one, d_i)))  # 1 x N - 1
            sum_exp_i = tf.reshape(tf.reduce_sum(exp_i, 1), shape=[]) # 0-D
            h_i = tf.reshape(tf.log(sum_exp_i) + tf.mul(tf.div(one, sum_exp_i), tf.scalar_mul(beta, tf.reduce_sum(tf.mul(d_i, exp_i), reduction_indices=1))), shape=[])
            p_i = tf.scalar_mul(tf.mul(one, sum_exp_i), exp_i)
            h_diff = tf.sub(h_i, logU)

            sess = tf.Session(graph=graph)
            init_op = tf.initialize_all_variables()
            sess.run(init_op)

            for b in range(FLAGS.binary_iterations):

                h_diff_ = sess.run(h_diff)

                if h_diff_ > 0:
                    beta_lim_ops = tf.assign(beta_min, beta)
                else:
                    beta_lim_ops = tf.assign(beta_max, beta)

                sess.run(beta_lim_ops)

                if h_diff_ > 0:
                    beta_max_ = sess.run(beta_max)
                    if beta_max_ == np.inf or beta_max_ == -np.inf:
                        new_beta = tf.mul(two, beta)
                    else:
                        new_beta = tf.div(tf.add(beta, beta_max), two)
                else:
                    beta_min_ = sess.run(beta_min)
                    if beta_min_ == np.inf or beta_min_ == -np.inf:
                        new_beta = tf.div(beta, two)
                    else:
                        new_beta = tf.div(tf.add(beta, beta_min), two)

                beta_ops = tf.assign(beta, new_beta)
                sess.run(beta_ops)

                if np.abs(h_diff_) < tol:
                    break

            P_[i, np.concatenate((np.r_[0:i], np.r_[i+1:no_rows]))] = sess.run(p_i)
            beta_[i] = sess.run(beta)
            sess.close()
            if i % 500 == 0:
                duration = time.time() - start_time
                print("Computed P-values for point ", i, " of ", no_rows, "(%.3f sec)"%duration)
                print("Mean value of sigma: {}".format(np.mean(np.sqrt(1 / beta_[0:i+1]))))
                start_time = time.time()

    print("Mean value of sigma: {}".format(np.mean(np.sqrt(1 / beta_))))

    return P_


if __name__ == "__main__":

    print("Run Y = tsne(X, no_rows, no_cols) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")

    X = np.loadtxt("mnist2500_X.txt")

    # Calculate Distance Matrix
    graph = tf.Graph()
    D = distance_matrix(graph, X)

    sess = tf.Session(graph=graph)
    D_ = sess.run(D)
    sess.close()

    P_ = D2P(D_)

    Y_ = P2Y(P_)
    print(Y_)
    # labels = np.loadtxt("mnist2500_labels.txt")
    # print(Y)
    # Plot.scatter(Y[:,0], Y[:,1], 20, labels)
    # Plot.show()