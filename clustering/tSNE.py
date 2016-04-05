import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np

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


def distance_matrix(graph, X):
    """
    :param X: Input Matrix for which to calculate Distance Matrix
    :return:
    """
    with graph.as_default():
        shape = X.get_shape().as_list()
        Di = tf.reduce_sum(tf.mul(X, X), 1)
        I_magnitude_tiled = tf.tile(tf.expand_dims(Di, 1), multiples=[1, shape[0]])
        I_t_magnitude_tiled = tf.tile(tf.reshape(Di, shape=[1, shape[0]]), multiples=[shape[0], 1])
        I_I = tf.matmul(X, tf.transpose(X))
        D = tf.sub(
            tf.add(I_magnitude_tiled, I_t_magnitude_tiled),
            tf.scalar_mul(2.0, I_I)
        )

    return D


def probability_matrix(graph, beta, D):
    """
    :param D:
    :return:
    """
    with graph.as_default():
        shape = D.get_shape().as_list()
        beta_matrix = tf.tile(tf.expand_dims(beta, 1), multiples=[1, shape[1]])
        sqrt_matrix = tf.tile(tf.constant(0.5, dtype=tf.float32, shape=[1,1]), multiples=[shape[0], shape[1]])
        zero_diagonal = tf.exp(tf.diag(diagonal=[-np.inf for _ in range(shape[0])]))
        exp = tf.mul(tf.exp(tf.mul(beta_matrix, tf.scalar_mul(-1.0, tf.pow(D, sqrt_matrix)))), zero_diagonal)
        sum_exp = tf.reduce_sum(exp, reduction_indices=1)
        sum_exp_matrix = tf.tile(tf.expand_dims(sum_exp, 1), multiples=[1, shape[1]])
        return tf.div(exp, sum_exp_matrix)


def entropy_matrix(graph, P):
    """

    :param graph:
    :param P:
    :return:
    """
    with graph.as_default():
        shape = P.get_shape().as_list()
        one_diagonal = tf.diag(diagonal=[1.0 for _ in range(shape[0])])
        P_mod = tf.add(P, one_diagonal)
        H = tf.reduce_sum(tf.scalar_mul(-1.0, tf.mul(P, tf.log(P_mod))), 1)
    return H


def training(input_data, perplexity=20.0):
    shape = input_data.shape

    X = tf.placeholder(dtype=tf.float32, shape=[shape[0], shape[1]])
    beta = tf.Variable(initial_value=tf.ones(shape=[shape[0]], dtype=tf.float32), name="beta")
    D = distance_matrix(tf.get_default_graph(), X)
    P = probability_matrix(tf.get_default_graph(), beta, D)
    H = entropy_matrix(tf.get_default_graph(), P)
    logU = tf.tile(tf.constant(np.log(perplexity), dtype=tf.float32, shape = [1]), multiples=[shape[0]])
    loss = tf.nn.l2_loss(H - logU, name='l2_Loss')

    beta_hist = tf.histogram_summary("beta", beta)
    loss_summary = tf.scalar_summary("l2_loss", loss)
    merged_summary = tf.merge_all_summaries()


    optimizer = tf.train.AdagradOptimizer(learning_rate=0.3)
    train_ops = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter("log/", sess.graph_def)
    beta_values = []
    loss_values = []
    start_time = time.time()

    for i in range(100):
        _, summary_str, loss_value, beta_value = sess.run([train_ops, merged_summary, loss, beta], feed_dict={X: input_data})
        beta_values.append(np.sum(beta_value)/len(beta_value))
        loss_values.append(loss_value)
        summary_writer.add_summary(summary_str, i)
        # Write the summaries and print an overview fairly often.
        if i % 25 == 0:
            # Print status to stdout.
            duration = time.time() - start_time
            start_time = time.time()
            print('Batch %d: (%.3f sec)' % (i, duration))
            print('L2-Loss')
            print(loss_value)
            print('')
            print('Beta Value')
            print(beta_value)
            print('')

    #Close Summary Writer
    summary_writer.close()
    # Expected Mean value of sigma:  2.38659662134 from original code
    fig, ax1 = plt.subplots()
    x = range(0, len(beta_values))
    ax1.plot(x, beta_values, 'r')
    ax1.axhline(y=2.38659662134, linewidth=2, color='g')
    ax1.axhline(y=beta_values[-1], linewidth=2, color='b')
    ax1.set_xlabel('#Epochs')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('mean-beta', color='r')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')

    ax2 = ax1.twinx()
    ax2.plot(x, loss_values, 'k', marker='*')
    ax2.set_ylabel('L2-Loss', color='k')
    for tl in ax2.get_yticklabels():
        tl.set_color('k')
    plt.show()

if __name__ == "__main__":

    X = np.loadtxt("mnist2500_X.txt")

    # Calculate Distance Matrix
    training(X)