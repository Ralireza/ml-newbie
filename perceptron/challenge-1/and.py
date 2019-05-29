import numpy as np
import perceptron.PERCEPTRON as p

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([1, 0, 0, 0])

perceptron = p.Perceptron(2)
perceptron.train(training_inputs, labels)

inputs = np.array([1, 1])
print(perceptron.predict(inputs))

inputs = np.array([0, 1])
print(perceptron.predict(inputs))
