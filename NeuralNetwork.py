import numpy as np

# Initialize network
def initialize_network(input_size, hidden_size, output_size):
    np.random.seed(42)
    network = {
        'W1': np.random.randn(input_size, hidden_size),
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size),
        'b2': np.zeros((1, output_size))
    }
    return network

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Forward propagation
def forward_propagation(network, X):
    W1, b1 = network['W1'], network['b1']
    W2, b2 = network['W2'], network['b2']

    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    cache = (z1, a1, z2, a2)
    return a2, cache

# Loss function
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Backpropagation
def backward_propagation(network, cache, X, y):
    W1, b1 = network['W1'], network['b1']
    W2, b2 = network['W2'], network['b2']
    z1, a1, z2, a2 = cache

    output_error = y - a2
    output_delta = output_error * sigmoid_derivative(a2)

    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(a1)

    dW2 = np.dot(a1.T, output_delta)
    db2 = np.sum(output_delta, axis=0, keepdims=True)

    dW1 = np.dot(X.T, hidden_delta)
    db1 = np.sum(hidden_delta, axis=0, keepdims=True)

    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return gradients

# Update parameters
def update_parameters(network, gradients, learning_rate):
    network['W1'] += learning_rate * gradients['dW1']
    network['b1'] += learning_rate * gradients['db1']
    network['W2'] += learning_rate * gradients['dW2']
    network['b2'] += learning_rate * gradients['db2']
    return network

# Train the network
def train_network(X, y, input_size, hidden_size, output_size, epochs, learning_rate):
    network = initialize_network(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        y_pred, cache = forward_propagation(network, X)

        loss = compute_loss(y, y_pred)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

        gradients = backward_propagation(network, cache, X, y)

        network = update_parameters(network, gradients, learning_rate)

    return network

# Predict
def predict(network, X):
    y_pred, _ = forward_propagation(network, X)
    return y_pred

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 2
output_size = 1
epochs = 10000
learning_rate = 0.1

trained_network = train_network(X, y, input_size, hidden_size, output_size, epochs, learning_rate)

predictions = predict(trained_network, X)
print(f'Predictions: \n{predictions}')
