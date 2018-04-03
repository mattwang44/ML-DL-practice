# Multi-Layer Perceptron on Iris Dataset

The structure built by MATLAB has 2 layers (single hidden layer with 3 neurons). Backpropagation is built without automation tool. Use only 3 features in Iris dataset.
L2 Loss is used and 0/1 error rate on test set is shown.

The results with different normalization methods and activation functions are saved in the folder named by specific learning rate and number of epoches (e.g. lr0.01epoch2500).

Here explains the alias of the normalization:
 * "none": no normalization is applied
 * "np1": each features are normalized in the range between -1 & +1
 * "p1": each features are normalized in the range between 0 & +1

![][2]

[2]: ./i.jpg