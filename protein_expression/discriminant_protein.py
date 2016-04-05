import tensorflow as tf
import numpy as np
import input_protein_data
import time

protein_expr = input_protein_data.read_data_sets("Protein_data")

input_vector_size = 77
no_of_classes = 8

# Symbolic Variable to store the Input. We feed in Input Later.
# x is a 2-D tensor of floating-point numbers, with a shape [None, 77].
# Here None means that a dimension can be of any length.
x = tf.placeholder(tf.float32, [None, input_vector_size])

# A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations
# It can be used and even modified by the computation. One generally has the model parameters be Variables

# Here each column_i in the 2-D Tensor is the weight associated with each input dimension (784) for the class = i
W = tf.Variable(tf.zeros([input_vector_size, no_of_classes]))
# This is the bias associated for each class_i where i is the row id
b = tf.Variable(tf.zeros([no_of_classes]))

# We defined a new tensor y, and by applying softmax transformation on the equation x*W + b
y = tf.nn.softmax(tf.matmul(x, W) + b)

# We defined a new tensor y_, we feed in the training labels in it.
y_ = tf.placeholder(tf.float32, [None, no_of_classes])

# We define Cross-Entropy as the Error function
# Cross-Entropy : the average length of communicating an event from one distribution with the
# optimal code for another distribution
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Using a Gradient Descent Optimizer with step size 0.01 and minimizing cross_entropy
optimizer = tf.train.GradientDescentOptimizer(1e-10)
train_op = optimizer.minimize(cross_entropy)

# Task to initialize all model variables
init = tf.initialize_all_variables()

# Start a TensorFlow session and initialize all model variables
sess = tf.Session()
sess.run(init)

# Training with Stochastic Gradient Descent Optimization
for i in range(10):
    batch_xs, batch_ys = protein_expr.next_batch(100)
    feed_dict = {x: batch_xs, y_: batch_ys}
    start_time = time.time()
    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `cross_entropy` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    _, cross_entropy_value = sess.run([train_op, cross_entropy],
                           feed_dict=feed_dict)

    duration = time.time() - start_time

    # Write the summaries and print an overview fairly often.
    if i % 2 == 0:
        # Print status to stdout.
        print('Batch %d: Cross Entropy = %.2f (%.3f sec)' % (i, cross_entropy_value, duration))

# Evaluate Values of the Weight Variable
weights = W.eval(sess)

# For every two pairs of class labels
# Find a subset of 3 proteins with maximum absolute difference in the weight attributed to their expression levels
# normalized by their total weight
for i in range(0, weights.shape[1]-1):
  for j in range(i+1, weights.shape[1]):
    class_profile_1 = weights[: , i]
    class_profile_2 = weights[: , j]
    difference = np.abs((class_profile_1-class_profile_2)/(class_profile_1+class_profile_2))
    # Results
    print("For classes : " + protein_expr.label_name[i] +" and " + protein_expr.label_name[j])
    print(protein_expr.label_name[i] + " +ve Discriminatory Proteins are : ",
          [protein_expr.feature_name[x] for x in np.argpartition(class_profile_1, -3)[-10:]])
    print(protein_expr.label_name[j] + " +ve Discriminatory Proteins are : ",
          [protein_expr.feature_name[x] for x in np.argpartition(class_profile_2, -3)[-10:]])
    print(protein_expr.label_name[i] + " -ve Discriminatory Proteins are : ",
          [protein_expr.feature_name[x] for x in np.argpartition(class_profile_1, -3)[0:10]])
    print(protein_expr.label_name[j] + " -ve Discriminatory Proteins are : ",
          [protein_expr.feature_name[x] for x in np.argpartition(class_profile_2, -3)[0:10]])
    print("Discriminator Proteins are : ", [protein_expr.feature_name[x] for x in  np.argpartition(difference, -3)[-3:]])