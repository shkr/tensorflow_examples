###Execute:

`python protein_expression/discriminant_protein.py`

###Problem Description

-----------

####Classes:

c-CS-s: control mice, stimulated to learn, injected with saline (9 mice)
c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice)
c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)
c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice)

t-CS-s: trisomy mice, stimulated to learn, injected with saline (7 mice)
t-CS-m: trisomy mice, stimulated to learn, injected with memantine (9 mice)
t-SC-s: trisomy mice, not stimulated to learn, injected with saline (9 mice)
t-SC-m: trisomy mice, not stimulated to learn, injected with memantine (9 mice)

The aim is to identify subsets of proteins that are discriminant between the classes.

####Reference : 

https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression#




###Solution
-----------

We defined a tensor y, which is the probability of a given class provided the protein expression profile (x),
and then apply softmax transformation on the equation 

```
y = soft_max( x*W + b )
```

We defined a new tensor `y_`, we feed in the training labels in it.

We define Cross-Entropy as the Error function.

In context of information theory it is formally defined as average length of communicating an event from one distribution with the
optimal code for another distribution

`cross_entropy = reduce(lambda v1,v2: v1 + v2, y_*log(y))`

We use a Stochastic Gradient Descent Optimization with step size `10e-4` and
minimizing cross_entropy to train the Weights & Bias.

The batch size used is `100`.

For every two pairs of class labels
Find a subset of 3 proteins with maximum absolute difference in the weight attributed to their expression levels
normalized by their total weight :

```
  | protein_profile_i - protein_profile_j | / | protein_profile_i + protein_profile_j |
```

Here protein_profile_i is a vector of size 77. 
Here each value is the weight associated with that protein and the label i.
The top 3 protein indexes with maximum values are selected as the subset.

