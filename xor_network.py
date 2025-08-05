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
    
    def backward(self, X, y, learning_rate=1.0):
        """Backward pass - this is where learning happens!"""
        m = X.shape[0]  # Number of training examples
        
        # Calculate loss (how wrong we are)
        loss = np.mean((self.a2 - y) ** 2)  # Mean squared error
        
        # BACKPROPAGATION MAGIC STARTS HERE
        
        # Step 1: Calculate error at output layer
        output_error = self.a2 - y  # How wrong is our final prediction?
        output_delta = output_error * self.sigmoid_derivative(self.a2)
        
        # Step 2: Calculate error at hidden layer
        # "How much did each hidden neuron contribute to the error?"
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Step 3: Update weights and biases
        # Move weights in direction that reduces error
        self.W2 -= learning_rate * self.a1.T.dot(output_delta) / m
        self.b2 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m
        
        self.W1 -= learning_rate * X.T.dot(hidden_delta) / m
        self.b1 -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m
        
        return loss
    
    def train(self, X, y, epochs=10000, learning_rate=1.0, print_every=1000):
        """Train the network using backpropagation"""
        losses = []
        
        print(f"Training for {epochs} epochs...")
        print("Epoch | Loss     | Predictions")
        print("------|----------|------------")
        
        for epoch in range(epochs):
            # Forward pass
            self.forward(X)
            
            # Backward pass (learning)
            loss = self.backward(X, y, learning_rate)
            losses.append(loss)
            
            # Print progress
            if epoch % print_every == 0 or epoch == epochs - 1:
                predictions = self.predict(X).flatten()
                print(f"{epoch:5d} | {loss:.6f} | {predictions}")
        
        return losses

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
    
    # NOW LET'S TRAIN IT!
    print("\n" + "="*50)
    print("TRAINING THE NETWORK TO LEARN XOR!")
    print("="*50)
    
    losses = nn.train(X, y, epochs=5000, learning_rate=10, print_every=500)
    
    # Test the trained network
    print("\n--- Testing Trained Network ---")
    final_predictions = nn.forward(X)
    print("Raw outputs:", final_predictions.flatten())
    print("Final predictions:", nn.predict(X).flatten())
    print("Expected:", y.flatten())
    print("Accuracy:", np.mean(nn.predict(X) == y) * 100, "%")
    
    # Plot the learning curve
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Show final decision boundary
    plt.subplot(1, 2, 2)
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.forward(grid_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Output')
    
    # Plot XOR points
    colors = ['red', 'blue', 'blue', 'red']  # 0s are red, 1s are blue
    for i, (point, color) in enumerate(zip(X, colors)):
        plt.scatter(point[0], point[1], c=color, s=100, edgecolor='black', linewidth=2)
        plt.annotate(f'({point[0]},{point[1]})→{y[i][0]}', 
                    (point[0], point[1]), xytext=(5, 5), textcoords='offset points')
    
    plt.title('XOR Decision Boundary')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("CONGRATULATIONS! Your network learned XOR! 🎉")
    print("="*50)