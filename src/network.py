import numpy as np

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def foward(self, a):
        for b, w, in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return sigmoid(a)
        

def sigmoid(a):
    return (1.0 / (1.0 + np.exp))
        
def output(network):
    print("=" * 50)
    print("Network view")
    print("=" * 50)
    
