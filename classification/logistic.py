import tensorflow.python.platform
import tensorflow as tf
from sklearn.datasets import make_classification, make_moons, make_circles
import numpy as np
import pylab as Plot

# Global variables.
BATCH_SIZE = 100  # The number of training examples to use per training step.
TEST_RATIO = 0.2 # The number of samples used to test
LEARNING_RATE = 0.1 # The learning rate for the gradient descent optimizer

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('data_set', None,
                           'This can be either linear, moon or circles.')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of epochs')
tf.app.flags.DEFINE_integer('num_samples', 1000,
                            'Number of examples')
FLAGS = tf.app.flags.FLAGS

data_sets = {
        "linear": make_classification(n_samples= FLAGS.num_samples, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1),
        "moon": make_moons(n_samples= FLAGS.num_samples, noise=0.3, random_state=1),
        "circles": make_circles(n_samples= FLAGS.num_samples, noise=0.2, factor=0.5, random_state=1)
    }


def main(argv=None):

    data_set = data_sets[FLAGS.data_set]
    (vectors, labels) = data_set

    (train_vectors, test_vectors) = \
        (vectors[: -int(vectors.shape[0] * TEST_RATIO), :], vectors[int(vectors.shape[0] * TEST_RATIO):, :])
    (train_labels, test_labels) = \
        (labels[: -int(vectors.shape[0] * TEST_RATIO)], labels[int(vectors.shape[0] * TEST_RATIO):])

    (train_size, num_features) = train_vectors.shape
    (test_size, num_classes) = (test_labels.size, max(np.max(test_labels), np.max(train_labels)) + 1)

    train_labels_one_hot = (np.arange(num_classes) == train_labels[:, None]).astype(np.float32)
    test_labels_one_hot = (np.arange(num_classes) == test_labels[:, None]).astype(np.float32)

    x = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
    y_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, num_features])

    # Matrix Operation
    W = tf.Variable(tf.zeros(shape=[num_features, num_classes]))
    b = tf.Variable(tf.zeros(shape=[num_classes]))
    y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

    # Cost Function
    cross_entropy = -tf.reduce_sum(tf.mul(y_placeholder, tf.log(y)))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_ops = optimizer.minimize(cross_entropy)

    # Accuracy
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    # Create Session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Epochs
    num_epochs = FLAGS.num_epochs

    # Summary
    cross_entropy_summary = tf.scalar_summary("cross_entropy", cross_entropy)
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)
    merged_summary = tf.merge_all_summaries()

    # Create Summary Writer for monitoring when training
    summary_writer = tf.train.SummaryWriter("logistic_log/", sess.graph_def)

    # Training
    for step in range(num_epochs * train_size // BATCH_SIZE):
        offset = (step * BATCH_SIZE) % train_size
        x_feed = train_vectors[offset: (offset + BATCH_SIZE), :]
        y_feed = train_labels_one_hot[offset: (offset + BATCH_SIZE), :]
        _, summary_str = sess.run([train_ops, merged_summary], feed_dict={x: x_feed, y_placeholder: y_feed})
        # Add Summary
        summary_writer.add_summary(summary_str, step)

    # Close Summary Writer
    summary_writer.close()

    # Test Accuracy
    test_accuracy_ = sess.run(accuracy, feed_dict={x: test_vectors, y_placeholder: test_labels_one_hot})
    print("Test Accuracy = {}".format(test_accuracy_))

    # Plot the test vectors w/ labels
    Plot.scatter(test_vectors[:, 0], test_vectors[:, 1], 20, (np.arange(3) == test_labels[:, None]).astype(np.float32))

    # Plot the hyperplane
    W_ = sess.run(W)
    b_ = sess.run(b)

    x_val = np.arange(np.min(test_vectors[:, 0]), np.max(test_vectors[:, 1]), 0.1)
    y_val = np.array([-1*(W_[0, 0]/W_[1, 0])*e + b_[0] for e in x_val])
    Plot.plot(x_val, y_val, linestyle='--', linewidth=2)

    Plot.title("Logistic Classifier w/ Softmax")
    Plot.show()


if __name__=='__main__':
    tf.app.run()