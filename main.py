import network
import numpy as np

def main():
    sizes = [784, 100, 10]
    print(f"Creating network with architecture: {sizes}")
    
    deepNeuron = network.Network(sizes)
    
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