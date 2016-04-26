import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import pylab as Plot

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

ENTROPY_CHECKPOINT_PATH = "save/entropy"
KL_DIVERGENCE_CHECKPOINT_PATH = "save/divergence"

def embedded_probability_matrix(graph, Y):
    """

    :param graph:
    :param Y:
    :return:
    """
    shape=Y.get_shape().as_list()
    Y_D = distance_matrix(graph, Y)
    ones = tf.ones(shape=[shape[0], shape[0]], dtype=tf.float32)
    zero_diagonal = tf.exp(tf.diag(diagonal=[-np.inf for _ in range(shape[0])]))
    power_curve = tf.add(ones, tf.mul(Y_D, Y_D))
    inverse_power_curve = tf.mul(zero_diagonal, tf.div(ones, power_curve))
    total_sum = tf.reduce_sum(inverse_power_curve, reduction_indices=[0, 1])
    total_sum_matrix = tf.tile(tf.reshape(total_sum, shape=[1,1]), multiples=[shape[0], shape[0]])
    Q = tf.div(inverse_power_curve, total_sum_matrix)
    return Q


def kl_divergence(graph, P, Q):
    """

    :param graph:
    :param P:
    :param Q:
    :return:
    """
    with graph.as_default():
        shape = P.get_shape().as_list()
        one_diagonal = tf.diag(diagonal=[1.0 for _ in range(shape[0])])
        P_mod = tf.add(P, one_diagonal)
        Q_mod = tf.add(Q, one_diagonal)
        return tf.reduce_sum(tf.mul(P, tf.log(tf.div(P_mod, Q_mod))), reduction_indices=[0, 1], name="kl_divergence")


def distance_matrix(graph, X):
    """
    :param X: Input Matrix for which to calculate Distance Matrix
    :return:
    """
    with graph.as_default():
        shape = X.get_shape().as_list()
        Di = tf.reduce_sum(tf.mul(X, X), 1)
        I_magnitude_tiled = tf.tile(tf.expand_dims(Di, 1), multiples=[1, shape[0]])
        I_t_magnitude_tiled = tf.tile(tf.expand_dims(Di, 0), multiples=[shape[0], 1])
        I_I = tf.matmul(X, tf.transpose(X))
        D = tf.add(
            tf.add(I_magnitude_tiled, I_t_magnitude_tiled),
            tf.scalar_mul(-2.0, I_I)
        )
        zero_diagonal = tf.add(
            tf.diag(diagonal=[-1.0 for _ in range(shape[0])]),
            tf.constant(value=1.0, dtype=tf.float32, shape=[shape[0], shape[0]])
        )
    return tf.mul(D, zero_diagonal)


def probability_matrix(graph, beta, D):
    """
    :param D:
    :return:
    """
    with graph.as_default():
        shape = D.get_shape().as_list()
        beta_matrix = tf.tile(tf.expand_dims(beta, 1), multiples=[1, shape[1]])

        zero_diagonal = tf.add(
            tf.diag(diagonal=[-1.0 for _ in range(shape[0])]),
            tf.constant(value=1.0, shape=[shape[0], shape[1]])
        )
        exp = tf.mul(
            tf.exp(tf.mul(beta_matrix, tf.scalar_mul(-1.0, D))),
            zero_diagonal
        )
        sum_exp = tf.maximum(tf.reduce_sum(exp, reduction_indices=1), tf.constant(-1.0, shape=[shape[0]]))

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


def training(input_data, perplexity, output_dimensions):
    """

    :param graph:
    :param input_data:
    :param perplexity:
    :return:
    """
    shape = input_data.shape
    print("X shape = ({}, {})".format(shape[0], shape[1]))
    # Input Data
    X = tf.placeholder(dtype=tf.float32, shape=[shape[0], shape[1]])
    beta = tf.Variable(initial_value=tf.constant(value=1.0, shape=[shape[0]], dtype=tf.float32), name="beta")
    D = distance_matrix(tf.get_default_graph(), X)
    P = probability_matrix(tf.get_default_graph(), beta, D)
    H = entropy_matrix(tf.get_default_graph(), P)

    # The perplexity of the distribution approximated around each input vector
    # must approach the logU value
    logU = tf.tile(tf.constant(np.log(perplexity), dtype=tf.float32, shape=[1]), multiples=[shape[0]])
    entropy_difference = tf.reduce_sum(H - logU, reduction_indices=0)
    entropy_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    entropy_train_ops = entropy_optimizer.minimize(entropy_difference)

    # Embedding
    # Y = tf.Variable(initial_value=tf.random_normal(shape=[shape[0], output_dimensions], ), name="Y")
    Y = tf.Variable(initial_value=tf.random_normal(shape=(shape[0], output_dimensions), dtype=tf.float32), name="Y")
    Q = embedded_probability_matrix(tf.get_default_graph(), Y)

    # KL-Divergence between the input probability matrix
    # and the embedded probability matrix
    embedding_loss = kl_divergence(tf.get_default_graph(), P, Q)
    embedding_optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    embedding_train_ops = embedding_optimizer.minimize(embedding_loss)

    # Summary
    beta_hist = tf.histogram_summary("beta", beta)
    beta_summary = tf.histogram_summary("beta_mean", tf.reduce_mean(beta, reduction_indices=0))
    entropy_loss_summary = tf.scalar_summary("entropy_difference", entropy_difference)
    embedding_loss_summary = tf.scalar_summary("kl_divergence", embedding_loss)
    merged_summary = tf.merge_all_summaries()

    # Create a Session to run Executions on the Graph
    sess = tf.Session()

    # Summary Writer
    summary_writer = tf.train.SummaryWriter("log/", sess.graph_def)

    # Create a Saver to check point the model after first training
    saver = tf.train.Saver()

    if os.path.exists(ENTROPY_CHECKPOINT_PATH):
        saver.restore(sess, ENTROPY_CHECKPOINT_PATH)
        print("Restored from Entropy Loss checkpoint")
    else:
        sess.run(tf.initialize_all_variables())
        beta_values = []
        loss_values = []
        start_time = time.time()
        for i in range(50):
            _, summary_str, loss_value, beta_value, d_value, p_value = sess.run([entropy_train_ops, merged_summary, entropy_difference, beta, D, P], feed_dict={X: input_data})
            print(p_value)
            beta_values.append(np.mean(beta_value))
            loss_values.append(loss_value)
            summary_writer.add_summary(summary_str, i)
            # Write the summaries and print an overview fairly often.
            if i % 1 == 0:
                # Print status to stdout.
                duration = time.time() - start_time
                start_time = time.time()
                print('Epoch %d: (%.3f sec)' % (i, duration))
                print('L2-Loss')
                print(loss_value)
                print('')
                print('Beta Value')
                print(beta_value)
                print('')

        # Expected Mean value of beta:  0.197530064535 from original code
        fig, ax1 = plt.subplots()
        x = range(0, len(beta_values))
        ax1.plot(x, beta_values, 'r')
        ax1.axhline(y=0.197530064535, linewidth=2, color='g')
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

        # Save at Entropy Checkpoint, useful when optimizing hyperparameters for training
        # the embedding Y
        # saver.save(sess, ENTROPY_CHECKPOINT_PATH)

    print("Using Beta With Mean Value = {}".format(sess.run(tf.reduce_mean(beta))))

    kl_divergence_values = []
    start_time = time.time()
    print('Y')
    print(sess.run(Y))
    print('')
    print('Q')
    print(sess.run(Q))
    print('')

    for i in range(300):
        _, summary_str, loss_value, Y_value = sess.run(
            [embedding_train_ops, merged_summary, embedding_loss, Y], feed_dict={X: input_data}
        )
        summary_writer.add_summary(summary_str, i)
        kl_divergence_values.append(loss_value)

        # Write the summaries and print an overview fairly often.
        if i % 50 == 0:
            # Print status to stdout.
            duration = time.time() - start_time
            start_time = time.time()
            print('Epoch %d: (%.3f sec)' % (i, duration))
            print('Differential Entropy')
            print(loss_value)
            print('')
            print('Y')
            print(Y_value)
            print('')


    #Close Summary Writer
    summary_writer.close()

    # Close Session
    Y_value = sess.run(Y)
    sess.close()

    # Plot the loss
    fig, ax1 = plt.subplots()
    x = range(0, len(kl_divergence_values))
    ax1.plot(x, kl_divergence_values, 'r')
    ax1.set_xlabel('#Differential Entropy')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('DE', color='r')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')
    plt.show()

    return Y_value

def pca(X = np.array([]), no_dims = 50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape;
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:,0:no_dims])
    return Y.real

if __name__ == "__main__":

    X = np.loadtxt("mnist2500_X.txt")
    # Set 0 as the mean
    (n, d) = X.shape

    # Calculate the embedding
    Y_value = training(pca(X), perplexity=20.0, output_dimensions=2)
    labels = np.loadtxt("mnist2500_labels.txt")
    Plot.scatter(Y_value[:,0], Y_value[:, 1], 20, labels)
    Plot.show()