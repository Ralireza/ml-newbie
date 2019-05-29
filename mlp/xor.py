from mlp.MultiLayerPerceptron import NueralNetwork as mlp
import numpy

# https://chih-ling-hsu.github.io/2017/08/30/NN-XOR#appendix
# https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d
# https://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php

numpy.random.seed(0)

# Set the input data
X = numpy.array([[0, 0], [0, 1],
                 [1, 0], [1, 1]])

# Initialize the NeuralNetwork with
# 2 input neurons
# 2 hidden neurons
# 1 output neuron
nn = mlp(X, [2, 2, 1])

# Set the labels, the correct results for the xor operation
y = numpy.array([0, 1,
                 1, 0])

# Call the fit function and train the network for a chosen number of epochs
nn.fit(X, y, epochs=10000)

# Show the prediction results
print("Final prediction")
for s in X:
    print(s, nn.predict_single_data(s))

nn.plot_decision_regions(X, y, nn)
