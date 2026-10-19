import numpy as np
from load_data import load_mnist

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    #here data comes in like z = [2.0, 1.0, 0.1]  
    
    #This is to get probablity of last layer output
    z = z - np.max(z, axis=0, keepdims=True)
    
    #this changes all numbers to postitive like [7.39, 2.72, 1.11] and then we can divide by sum to get probablity  
    e = np.exp(z)

    #this devides each value by total of sum
    #total = 7.39 + 2.72 + 1.11 = 11.22
    #7.39 / 11.22 = 0.66
    #2.72 / 11.22 = 0.24
    #1.11 / 11.22 = 0.10
    return e / np.sum(e, axis=0, keepdims=True)

class Network:
    def __init__(self, sizes, seed=42):
        #sizes is a list like [784,30,10]
        #784 imputs, 30 hideen neurons, 10 outputs
        rng = np.random.default_rng(seed)

        self.sizes = sizes
        self.num_layers = len(sizes)

        #One weight matrix and one bias vector per connection between layers
        #zip(sizes[:-1], sizes[1:]) pairs up (n_in, n_out):
        #[784, 30, 10], sizes[:-1] = every layer except the last, [784, 30]
        #[784, 30, 10], sizes[1:] = every layer except the first, [30, 10]
        #[784, 30, 10] -> (784, 30), (30, 10)
        #Zipping them gives each connection’s (n_in, n_out)
        self.weights = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.weights.append(rng.standard_normal((n_out, n_in)))

        self.biases = []
        for n_out in sizes[1:]:
            self.biases.append(np.zeros((n_out,1)))

    def count_parameters(self):
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total

    def feedforward(self, x):
        #x is a column vector of shape (784, 1).
        a = x
        
        #hidden laers = weighted sum + bias, then sigmoid.
        #[:-1] means all layers except the last one
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = sigmoid(w @ a +b)
        
        #output layer: weighted sum + bias, then softmax
        z = self.weights[-1] @ a + self.biases[-1]
        return softmax(z)

if __name__ == "__main__":
    training_data, _, _ = load_mnist()
    train_inputs, train_labels = training_data

    net = Network([784, 30, 10])
    
    image = train_inputs[0].reshape(784, 1)
    output = net.feedforward(image)

    print("Output shape:", output.shape)
    print("Probabilities:", output.ravel().round(3))
    print("Sum of probabilities:", output.sum())
    print("Predicted digit:", np.argmax(output))
    print("Actual digit:", train_labels[0])