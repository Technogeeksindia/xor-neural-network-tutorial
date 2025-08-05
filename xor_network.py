import numpy as np
import matplotlib.pyplot as plt

class XORNeuralNetwork:
    def __init__(self):
        # Network architecture: 2 inputs -> 3 hidden -> 1 output
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1
        
        # Initialize weights and biases randomly
        # Small random values help with learning
        np.random.seed(42)  # For reproducible results
        
        # Weights from input to hidden layer (2x3 matrix)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.5
        self.b1 = np.zeros((1, self.hidden_size))
        
        # Weights from hidden to output layer (3x1 matrix)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.5
        self.b2 = np.zeros((1, self.output_size))
        
        print("Neural Network initialized!")
        print(f"W1 shape: {self.W1.shape}, b1 shape: {self.b1.shape}")
        print(f"W2 shape: {self.W2.shape}, b2 shape: {self.b2.shape}")
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward pass through the network"""
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear combination
        self.a1 = self.sigmoid(self.z1)         # Apply activation
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear combination
        self.a2 = self.sigmoid(self.z2)               # Apply activation
        
        return self.a2
    
    def predict(self, X):
        """Make predictions (forward pass)"""
        output = self.forward(X)
        return (output > 0.5).astype(int)  # Convert to 0 or 1

# Let's test our network structure
if __name__ == "__main__":
    # Create XOR dataset
    X = np.array([[0, 0],
                  [0, 1], 
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1], 
                  [0]])
    
    print("XOR Dataset:")
    print("Inputs:", X)
    print("Expected outputs:", y.flatten())
    
    # Create and test network
    nn = XORNeuralNetwork()
    
    # Test forward pass with untrained network
    print("\n--- Testing Untrained Network ---")
    predictions = nn.forward(X)
    print("Raw outputs:", predictions.flatten())
    print("Predictions (>0.5):", nn.predict(X).flatten())
    print("Expected:", y.flatten())