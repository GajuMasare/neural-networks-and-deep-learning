import numpy as np
from load_data import load_mnist

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    #converts any number or array of numbers into range (0,1)

def neuron_output(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    # step 1: weighted sum. np.dot multiplies each input by its weight 
    # an adds eveything up

    return sigmoid(z)
    # step 2: bends the result into 0-1 range

if __name__ == '__main__':
    # test 1: hardcoded example to check
    inputs = np.array([1,0,1])
    weights = np.array([0.5, -0.3, 0.8])
    bias = 5

    # By hand: z = (1*0.5) + (0*-0.3) + (1*0.8) + 0.1 = 1.4
    # sigmoid(1.4) = 1 / (1 + e^-1.4) ≈ 0.8022
    print("Neuron output:", neuron_output(inputs, weights, bias))

    #test 2 : sigmoid behaviour
    for z in [-10, -2, 0, 2, 10]:
        print(f"sigmoid({z:>3}) = {sigmoid(z):.6f}")

    #test 3 : a nuron looking at a real MNIST image
    training_data, _, _ = load_mnist()
    train_inputs, train_labels = training_data

    image = train_inputs[0]  # first image in the training set (shape (784,))
    rng = np.random.default_rng(seed=42)  # seeded so repeated results
    weights = rng.standard_normal(784) # random weights for each pixel
    bias = 0.0 

    print("Label of this image:", train_labels[0])
    print("Random neuron's output:", neuron_output(image, weights, bias))
