Disclaimer: I take 0 credits for this project. It is a simple "Neural Network from scratch using Python" project I encountered on LinkedIn (including the full code) and copied it here.

# Simple Neural Network using Python
![Simple Neural Network](https://github.com/fabiogirones/simple-neural-network/assets/45832602/ff08d621-deed-4f69-93df-0d214872e882)

## Step 1: Structure of the Neural Network
A typical neural network consists of three types of layers:
- Input layer: receives the input data
- Hidden layer: perform computations and feature extraction
- Output layer: produces the final output
  
![Epochs Network](https://github.com/fabiogirones/simple-neural-network/assets/45832602/32ebe1a9-0380-42f1-914b-29a3756b641b)

## Step 2: Structure of the Neural Network
#### def initialize_network(input_size, hidden_size, output_size)
We initialize the network with random weights and biases.
Input: input size, hidden size, output size
Process: intialize weights (W1, W2), and biases (b1, b2) with random values

## Step 3: Activation functions
#### Sigmoid function: def sigmoid(x)
The sigmoid activation function is a commonly used activation function in neural networks. It maps any real-valued number into the range (0, 1). The function is defined as follows:
σ(x)=1/(1+e^-x)

#### Derivative function: def sigmoid_derivative(x)
The derivative of the sigmoid function is useful for backpropagation in neural networks. The derivative can be expressed in terms of the output of the sigmoid function itself:
σ′(x)=x⋅(1−x)

## Step 4: Forward propagation
#### def forward_propagation(network, X)
Forward propagation involves passing input data through the layers of the network to obtain an output. Each neuron in a layer performs a weighted sum of the inputs, applies an activation function, and passes the results to the next layer. Computes the output of the network by passing inputs through each layer.

## Step 5: Loss function
#### def compute_loss(y_true, y_pred)
The loss function measures how well the neural network's predictions match the actual data. Common loss functions include Mean Squared Error (MSE) for regression and Cross-Entropy for classification. Measures the error between the predicted and actual outputs. We'll use Mean Squared Error (MSE) for simplicity.
- Input: Actual output (y), predicted output (a2)
- Process: Compute mean squared error loss
- Output: loss value

## Step 6: Backpropagation
#### def backward_propagation(network, cache, X, y)
Backpropagation is the process of updating the network's weights to minimize the loss. It involves computing the gradient of the loss function with respect to each weight and adjusting the weights on the direction that reduces the loss.

## Step 7: Update weights
#### def update_parameters(network, gradients, learning_rate)
Adjusts the weights using computed gradients

## Step 8: Training function
#### train_network(x, y, input_size, hidden_size, output_size, epochs, learning_rate)
Train a neural network involves iteratively performing forward propagation, computing the loss, performing backpropagation, and updating the weights. Trains the network by performing forward and backward propagation and updating the weights iteratively.

## Step 9: Test the network
#### def predict(network, X)
Tests the network on sample input data
