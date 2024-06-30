Disclaimer: I take 0 credits for this project. It is a simple "Neural Network from scratch using Python" project I encountered on LinkedIn (including the full code) and copied it here.

# Simple Neural Network using Python
![Simple Neural Network](https://github.com/fabiogirones/simple-neural-network/assets/45832602/ff08d621-deed-4f69-93df-0d214872e882)

## Step 1: Structure of the Neural Network
A typical neural network consists of three types of layers:
- Input layer: receives the input data
- Hidden layer: perform computations and feature extraction
- Output layer: produces the final output
- 
![Epochs Network](https://github.com/fabiogirones/simple-neural-network/assets/45832602/32ebe1a9-0380-42f1-914b-29a3756b641b)

## Step 2: Structure of the Neural Network
def initialize_network
We initialize the network with random weights and biases
Input: input size, hidden size, output size
Process: intialize weights (W1, W2), and biases (b1, b2) with random values

## Step 3: Activation functions
### Sigmoid function
The sigmoid activation function is a commonly used activation function in neural networks. It maps any real-valued number into the range (0, 1). The function is defined as follows:
σ(x)=1/(1+e^-x)

### Derivative function
The derivative of the sigmoid function is useful for backpropagation in neural networks. The derivative can be expressed in terms of the output of the sigmoid function itself. If y is the output of the sigmoid function, then the derivative is:
σ′(x)=x⋅(1−x)
