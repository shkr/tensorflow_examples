import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('alpha', 0.3, 'Initial learning rate.')
flags.DEFINE_float('sigma', 10.0, 'Neighbourhood Gaussian Function Standard Deviation.')
flags.DEFINE_integer('max_steps', 400, 'Number of steps to run trainer.')
flags.DEFINE_integer('dimensions', 3, 'Dimension of Input Vector')
flags.DEFINE_integer('n_rows', 20, 'Number of rows in the Kohonen Network Rectangular lattice')
flags.DEFINE_integer('n_cols', 30, 'Number of columns in the Kohonen Network Rectangular lattice')

def node_locations(rows, cols):
    """
    Yields  2-D locations of the individual neurons in the SOM
    :param rows the number of rows in the rectangular lattice of the kohenen network
    :param cols the number of columns in the rectangular lattice of the kohenen network
    """
    #Nested iterations over both dimensions
    #to generate all 2-D locations in the map
    for i in range(rows):
        for j in range(cols):
            yield [i, j]


def build_network():
  """
  Here we build the Kohenon Network and return
  the pieces to the calling function.
  We need to create a data flow graph where
  nodes represent operations (ops), and the edges represent tensors.
  The entire dataflow graph is executed within a session.
  """

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():

    # D : FLAGS.dimensions
    # vector : 1 x D
    # Generate placeholders for input vectors
    vector_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.dimensions])
    iteration_number = tf.placeholder(tf.float32)

    # Build a Graph that computes predictions from the inference model.
    with tf.name_scope('som'):

        # N = FLAGS.n_rows * FLAGS.n_cols
        # node_loc : N x 2
        node_loc = tf.pack(list(node_locations(FLAGS.n_rows, FLAGS.n_cols)))

        # weights : N x D
        weights = tf.Variable(
            tf.truncated_normal([FLAGS.n_rows * FLAGS.n_cols, FLAGS.dimensions], stddev=1.0),
            name='weights'
        )

        # Matching Phase
        # The discriminant function here is the squared euclidean distance
        # vector_packed : N x D
        vector_packed = tf.pack([vector_placeholder for i in range(FLAGS.n_rows * FLAGS.n_cols)])
        # euclidean_distance : N x 1
        #      ReduceSumOverRows([(N x D) - (N x D)]**2) = N x 1
        #      (N x 1)**0.5
        euclidean_distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(weights, vector_packed), 2), 1))

        # BMU - Node Index
        bmu_index = tf.argmin(euclidean_distance, 0)

        slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
        bmu_loc = tf.reshape(tf.slice(node_loc, slice_input, tf.constant(np.array([1, 2]))), [2])

        #alpha and sigma values based on iteration number
        learning_rate_op = tf.sub(1.0, tf.div(iteration_number, FLAGS.max_steps))
        alpha_op = tf.mul(FLAGS.alpha, learning_rate_op)
        sigma_op = tf.mul(FLAGS.sigma, learning_rate_op)

        #Construct the op that will generate a vector with learning
        #rates for all neurons, based on iteration number and location wrt BMU.
        # distance_squares  = { node_loc - bmu_loc_{N x 2} }**2 : 1 x N
        distance_squares = tf.reduce_sum(tf.pow(tf.sub(node_loc, tf.pack([bmu_loc for i in range(FLAGS.n_rows * FLAGS.n_cols)])), 2), 1)
        # neighbourhood_func  : 1 x N
        neighbourhood_func = tf.exp(tf.neg(tf.div(tf.cast(distance_squares, "float32"), tf.pow(sigma_op, 2))))
        # # neighbourhood_func  : 1 x N
        learning_rate_op = tf.mul(alpha_op, neighbourhood_func)

        #Finally, the op that will use learning_rate_op to update
        #the weight vectors of all neurons based on a particular
        #learning_rate_multiplier : N x [ 1 x D] = N x D
        learning_rate_multiplier = tf.pack([tf.tile(tf.slice(learning_rate_op, np.array([i]), np.array([1])), [FLAGS.dimensions]) for i in range(FLAGS.n_rows * FLAGS.n_cols)])
        weights_delta = tf.mul(learning_rate_multiplier, tf.sub(tf.pack([vector_placeholder for i in range(FLAGS.n_rows * FLAGS.n_cols)]), weights))

        new_weights_op = tf.add(weights, weights_delta)
        training_op = tf.assign(weights, new_weights_op)

        ##INITIALIZE VARIABLES
        sess = tf.Session()
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

  return (sess, vector_placeholder, iteration_number, weights, node_loc, training_op)

def train(input_vects):
    """
    Trains the Kohenon Network with the input_vects
    'input_vects' should be an iterable of 1-D NumPy arrays with dimensionality as mentioned in the FLAGS.
    """
    (sess, vector_placeholder, iteration_number, weights, node_loc, training_op) = build_network()
    #Iteratively train with each vector
    for iter_no in range(FLAGS.max_steps):
        for input_vect in input_vects:
            sess.run(training_op, feed_dict={vector_placeholder: input_vect, iteration_number: iter_no})

    #Create a centroid grid
    centroid_grid = [[] for i in range(FLAGS.n_rows)]
    _weights = list(sess.run(weights))
    _node_loc = list(sess.run(node_loc))
    for i, loc in enumerate(_node_loc):
        centroid_grid[loc[0]].append(_weights[i])
    return _weights, _node_loc, centroid_grid


def map_vects(input_vects, test_vectors):
    """
    Maps each test vector to the closest node
    'input_vects' should be an iterable of 1-D NumPy arrays with dimensionality as mentioned in the FLAGS.
    :param input_vects input vectors
    :param test_vectors test vectors with labels
    """
    _weights, _node_loc, centroid_grid = train(input_vects)
    mapped = []
    for vect in test_vectors:
        min_index = min([i for i in range(len(_weights))], key=lambda x: np.linalg.norm(vect-_weights[x]))
        mapped.append(_node_loc[min_index])

    return centroid_grid, mapped

if __name__=='__main__':

    #Training input vectors which represent RGBcolors
    colors = np.array(
         [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])
    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']

    test_colors = np.array([[1.0, 0., 247.0/255.0], [1.0, 145.0/255.0, 0.0]], np.float32)

    test_color_names = ['pink', 'orange']

    # Map vectors to their closest nodes
    image_grid, mapped = map_vects(colors, test_colors)

    #Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], test_color_names[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()
