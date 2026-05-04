from src import mnist_loader
from src import network4
import numpy as np

def main():
    # sizes = [784, 10]
    # print(f"Creating network with architecture: {sizes}")
    
    # deepNeuron = network2.Network(sizes)
    
    # network2.output(deepNeuron)

    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # deepNeuron.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    # 1. Load the data using the loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # 2. Define architecture (784 inputs for MNIST pixels, 30 hidden, 10 outputs)
    sizes = [784, 30, 10]
    print(f"Creating network with architecture: {sizes}")
    
    # 3. Initialize the network
    # Note: Default cost is CrossEntropyCost in network2.py
    deepNeuron = network4.Network(sizes)

    if(deepNeuron.loaded_from_file == True):
        print(f"Already a network with the same architecture exists!")
        
    
    # 4. Initiate Stochastic Gradient Descent (SGD)
    # Parameters: data, epochs, mini_batch_size, learning_rate (eta), regularization (lmbda)
    # deepNeuron.SGD(
    #     training_data, 
    #     epochs=30, 
    #     mini_batch_size=10, 
    #     eta=0.5, 
    #     lmbda=5.0, 
    #     evaluation_data=test_data,
    #     monitor_evaluation_accuracy=True,
    #     monitor_evaluation_cost=True,
    #     monitor_training_accuracy=True,
    #     monitor_training_cost=True
    # )

    # deepNeuron.save("my_trained_network.json")

if __name__ == "__main__": 
    main()