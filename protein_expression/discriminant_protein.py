import tensorflow as tf
import input_protein_data

protein_expr = input_protein_data.read_data_sets("Protein_data")

input_vector_size = 77
no_of_classes = 8

v = tf.Variable([1, 2])

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
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

# Task to initialize all model variables
init = tf.initialize_all_variables()

# Start a TensorFlow session and initialize all model variables
sess = tf.Session()
sess.run(init)

# Training with Stochastic Gradient Descent Optimization
for i in range(10):
  batch_xs, batch_ys = protein_expr.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

import numpy as np

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
    print("Discriminator Proteins are : ", map(lambda x: protein_expr.feature_name[x], np.argpartition(difference, -3)[-3:]))



