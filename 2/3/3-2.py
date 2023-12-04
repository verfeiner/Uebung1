import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load CIFAR-10 data locally
def load_cifar10_local(root_folder):
    x_train, y_train, x_test, y_test = [], [], [], []

    for batch in range(1, 6):
        with open(os.path.join(root_folder, f'data_batch_{batch}'), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            x_train.append(data[b'data'])
            y_train.append(data[b'labels'])

    with open(os.path.join(root_folder, 'test_batch'), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        x_test.append(data[b'data'])
        y_test.append(data[b'labels'])

    return (
        np.concatenate(x_train), np.concatenate(y_train),
        np.concatenate(x_test), np.concatenate(y_test)
    )

# Specify the path to the CIFAR-10 dataset
path_to_cifar = 'C:/Users/Bin/Desktop/Master OBV/Master-WS/BALG/cifar-10-batches-py'  # Replace with your actual path

# Load the dataset
X_train, y_train, X_test, y_test = load_cifar10_local(path_to_cifar)

# Reshape and normalize pixel values
X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0

# Flatten each image in X_train and X_test
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# One-hot encode the labels
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

# Softmax Function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probabilities

# Cross-Entropy Loss Function
def cross_entropy_loss(probabilities, y):
    m = len(y)
    loss = -np.sum(np.log(probabilities[np.arange(m), y])) / m
    return loss

# Training Loop
learning_rates = [1e-7]#, 1e-6, 1e-5, 1e-4, 1e-3]
reg_strengths = [1e2]#, 1e4, 1e3, 1e2, 1e1]

for learning_rate in learning_rates:
    for reg_strength in reg_strengths:
        print(f"Training with learning rate: {learning_rate}, regularization strength: {reg_strength}")

        # Initialize weights and biases
        input_size = X_train_flat.shape[1]
        output_size = num_classes

        W = np.random.randn(input_size, output_size)
        b = np.zeros(output_size)

        # Hyperparameters
        num_epochs = 500
        batch_size = 32

        # Training Loop
        for epoch in range(num_epochs):
            for i in range(0, len(X_train_flat), batch_size):
                # Forward pass
                batch_X = X_train_flat[i:i+batch_size]
                batch_y = y_train_onehot[i:i+batch_size]

                logits = np.dot(batch_X, W) + b
                probabilities = softmax(logits)
                loss = cross_entropy_loss(probabilities, np.argmax(batch_y, axis=1))

                # Regularization term
                reg_term = 0.5 * reg_strength * np.sum(W**2)

                # Total loss
                total_loss = loss + reg_term

                # Backward pass
                gradient = probabilities
                gradient[np.arange(len(batch_y)), np.argmax(batch_y, axis=1)] -= 1
                gradient /= len(batch_X)

                dW = np.dot(batch_X.T, gradient) + reg_strength * W
                db = np.sum(gradient, axis=0)

                # Update weights and biases
                W -= learning_rate * dW
                b -= learning_rate * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss}")

        # Evaluate the model on the test set
        logits_test = np.dot(X_test_flat, W) + b
        probabilities_test = softmax(logits_test)
        predictions_test = np.argmax(probabilities_test, axis=1)

        accuracy = accuracy_score(y_test, predictions_test)
        print(f"Test Accuracy: {accuracy}")
        print("--------------")
