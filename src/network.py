import numpy as np
import random

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SDG(self, training_data, mini_batch_size, epoch, eta, test_data=None): # back propagation
        if(test_data):
            n_test = len(test_data)
        
        n = len(training_data)

        for j in range(epoch):
            random.shuffle(training_data)

            # mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            mini_batches = []
            for k in range(0, n, mini_batch_size):
                batch = training_data[k: k + mini_batch_size]
                mini_batches.append(batch)

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}").format(
                    j, self.evaluate(test_data), n_test)
            else:
                print("Epoch {0} complete").format(j)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def output(network):
    print("=" * 50)
    print("Network View")
    print("=" * 50)
    
    print(f"Network architecture: {network.sizes}")
    print(f"Number of layers: {network.num_layers}")
    print()
    
    print(f"Layer 0 (INPUT): {network.sizes[0]} neurons")
    print(f"  No weights or biases (input layer)")
    print()
    
    for i in range(1, network.num_layers):
        layer_type = "HIDDEN" if i < network.num_layers - 1 else "OUTPUT"
        
        neurons = network.sizes[i]
        bias_shape = network.biases[i-1].shape
        weight_shape = network.weights[i-1].shape
        prev_neurons = network.sizes[i-1]
        
        print(f"Layer {i} ({layer_type}): {neurons} neurons")
        print(f"  Bias shape: {bias_shape} ({neurons} bias values)")
        print(f"  Weight shape: {weight_shape} (connects {prev_neurons} → {neurons} neurons)")
        
        print(f"  Bias sample (first 3): {network.biases[i-1].flatten()[:3]}")
        print(f"  Weight sample (first 3x3):\n{network.weights[i-1][:3, :3]}")
        print()


def main():
    sizes = [2, 3, 67, 45, 1]
    print(f"Creating network with architecture: {sizes}")
    
    deepNeuron = Network(sizes)
    
    output(deepNeuron)
    
    print("\n" + "=" * 50)
    print("TESTING WITH SAMPLE DATA")
    print("=" * 50)
    
    test_inputs = [
        np.array([[0], [0]]),
        np.array([[0], [1]]),
        np.array([[1], [0]]),
        np.array([[1], [1]])
    ]
    
    for i, test_input in enumerate(test_inputs):
        output = deepNeuron.forward(test_input)
        print(f"Input {test_input.flatten()} → Output: {output[0][0]:.6f}")

if __name__ == "__main__": 
    main()