import gzip, pickle

def load_mnist():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        return training_data, test_data, validation_data 

if __name__ == '__main__':
    training_data, test_data, validation_data = load_mnist()

    # Each *_data is a tuple: (inputs, labels)
    train_inputs, train_labels = training_data 
    test_inputs, test_labels = test_data
    valid_inputs, valid_labels = validation_data


    # Sanity check: look at one label and one input's shape
    print("First Label:", train_labels[0])
    print("First input vector length:", train_inputs[0].shape)
