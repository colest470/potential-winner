from src import mnist_loader
from src import network4
import numpy as np

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    sizes = [784, 30, 10]
    print(f"Creating network with architecture: {sizes}")
    
    deepNeuron = network4.Network(sizes)

    if(deepNeuron.loaded_from_file == True):
        print(f"Already a network with the same architecture exists!")
        

if __name__ == "__main__": 
    main()