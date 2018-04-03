# Multi-Layer Perceptron on Iris Dataset

The structure built by MATLAB has 2 layers (single hidden layer with 3 neurons). Backpropagation is built without automation tool. Use only 3 features in Iris dataset.
L2-loss is used and 0/1 error rate on test set is shown.

There are four acttivation functions can be used in the hidden layer and the output layer:
 * tanh
 * sigmoid 
 * ReLU
 * Leaky ReLU

Here explains the alias of the normalization applied on the input data:
 * "none": no normalization is applied
 * "np1": each features are normalized in the range between -1 & +1
 * "p1": each features are normalized in the range between 0 & +1

![][2]

![][2]

[1]: ./lrelu-sigmoid.jpg
[2]: ./tanh-sigmoid.jpg