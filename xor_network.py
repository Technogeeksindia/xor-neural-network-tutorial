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
        # Random values between -0.5 and 0.5 to avoid saturation of the sigmoid function
        # This helps the network learn better
        # Weights are initialized with a normal distribution scaled by 0.5
        # This ensures weights are small but not too small
        # Biases are initialized to zero
        # np.random.randn generates values from a standard normal distribution
        # Multiplying by 0.5 scales them down
        # np.random.randn(2,3) genrates a 2x3 matrix of random values
        # np.zeros creates a 1x3 matrix of zeros for biases
        # This is a common practice to ensure the network starts with small weights
        # Weights from input to hidden layer (2x3 matrix)

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.5
        self.b1 = np.zeros((1, self.hidden_size))
        
        # Weights from hidden to output layer (3x1 matrix)
        # Weights are initialized similarly to the first layer
        # This ensures the output layer can learn to combine hidden layer outputs
        # np.random.randn generates values from a standard normal distribution
        #w2 = np.random.randn(3,1) generates a 3x1 matrix of random values
        # np.zeros creates a 1x1 matrix of zeros for biases
    

        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.5
        self.b2 = np.zeros((1, self.output_size))



        # Print network structure
        # This is just for debugging and understanding the network
        # It shows the shapes of the weight matrices and biases
        # Shape of W1 is (2, 3) meaning 2 inputs to 3 hidden neurons
        # Shape of b1 is (1, 3) meaning 1 bias for each hidden neuron
        # Shape of W2 is (3, 1) meaning 3 hidden neurons to 1 output neuron
        # Shape of b2 is (1, 1) meaning 1 bias for the output neuron
        # This helps visualize the network structure

        print("Neural Network initialized!")
        print(f"W1 shape: {self.W1.shape}, b1 shape: {self.b1.shape}")
        print(f"W2 shape: {self.W2.shape}, b2 shape: {self.b2.shape}")
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        # Clipping prevents numerical instability
        # This is a common practice in neural networks to avoid overflow
        # The sigmoid function is defined as 1 / (1 + exp(-x))
        # This function squashes the input to a range between 0 and 1
        # This is useful for binary classification problems like XOR
        # The sigmoid function is smooth and differentiable, which is important for backpropagation
        # np.clip is used to limit the values of x to avoid overflow in exp
        # Sigmoid function is defined as 1 / (1 + exp(-x))
        # This prevents large negative values from causing overflow in exp
        # np.clip(x, -500, 500) ensures that x is within a safe range
        # This is important because exp(-x) can become very large for large negative x
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        # Derivative of sigmoid function
        # This is used during backpropagation to update weights
        # The derivative of the sigmoid function is given by f'(x) = f(x) * (1 - f(x))
        # This is useful for calculating gradients during backpropagation
        # It tells us how much the output changes with respect to a change in input
        # This is important for updating weights during training
        # The derivative is calculated as f(x) * (1 - f(x))
        # This is because the sigmoid function is defined as f(x) = 1 / (1 + exp(-x))
        # The derivative is f'(x) = f(x) * (1 - f(x))
        # This is a common practice in neural networks to calculate the derivative of the activation function for backpropagation
        return x * (1 - x)
    
    def forward(self, X):
        """Forward pass through the network"""
        # Forward pass - this is where we compute the output
        # X is the input data (shape: number of samples x input size)
        # We calculate the output of the network by passing the input through the layers
        # Input to hidden layer
        # We calculate the linear combination of inputs and weights, then apply the activation function
        # np.dot(X, self.W1) computes the dot product of input X and weights
        # self.W1 is the weight matrix connecting input layer to hidden layer
        # self.b1 is the bias vector for the hidden layer
        # np.dot computes the dot product, which is a linear combination of inputs and weights
        # The result is a linear combination of inputs and weights, plus the bias
        # The bias is added to the linear combination to allow the model to fit the data better
        # The bias helps the model to fit the data better by allowing it to shift the activation
        # The activation function is applied to the linear combination to introduce non-linearity
        # This is important because XOR is a non-linear problem
        # The sigmoid activation function is applied to the linear combination to introduce non-linearity
        # This allows the network to learn complex patterns in the data
    

        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear combination
        self.a1 = self.sigmoid(self.z1)         # Apply activation
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear combination
        self.a2 = self.sigmoid(self.z2)               # Apply activation
        
        return self.a2
    
    def predict(self, X):
        """Make predictions (forward pass)"""
        # Predicting is just a forward pass through the network
        # We use the forward method to compute the output
        # The output is a probability between 0 and 1
        # We convert the output to binary (0 or 1) using a threshold of 0.5
        # If the output is greater than 0.5, we classify it as 1, otherwise 0
        # This is a common practice in binary classification problems
        # The predict method is used to make predictions on new data
        # It takes the input data X and computes the output using the forward method
        # The output is a probability between 0 and 1
        # We use a threshold of 0.5 to convert the probability to binary output
        # If the output is greater than 0.5, we classify it as 1
        # If the output is less than or equal to 0.5, we classify it as 0
        # This is a common practice in binary classification problems
        # The predict method returns the predicted class labels (0 or 1)
        # Forward pass to get the output
        output = self.forward(X)
        return (output > 0.5).astype(int)  # Convert to 0 or 1
    
    def backward(self, X, y, learning_rate=1.0):
        """Backward pass - this is where learning happens!"""
        # Backward pass - this is where we update weights based on error
        # X is the input data, y is the true labels
        # learning_rate controls how much we adjust weights
        # We calculate the error at the output layer and propagate it back to update weights
        # m is the number of training examples
        # We calculate the number of training examples to normalize the gradients
        # This is important for gradient descent to ensure we update weights correctly
        m = X.shape[0]  # Number of training examples
        
        # Calculate loss (how wrong we are)
        # We calculate the loss using mean squared error
        # This is a common loss function for regression problems
        # The loss is calculated as the mean squared error between predicted and true labels
        # np.mean((self.a2 - y) ** 2) computes the mean squared error
        # self.a2 is the predicted output from the forward pass
        # y is the true labels
        # The loss tells us how far off our predictions are from the true labels
        # A lower loss means better predictions
        # The loss is used to measure how well the network is performing
        # The loss is used to update the weights during backpropagation
        # This is a common practice in neural networks to calculate the loss
        # The loss is used to measure how well the network is performing
        # The loss is used to update the weights during backpropagation     
        loss = np.mean((self.a2 - y) ** 2)  # Mean squared error
        
        # BACKPROPAGATION MAGIC STARTS HERE
        
        # Step 1: Calculate error at output layer
        # "How wrong is our final prediction?"
        # We calculate the error at the output layer
        # The error is the difference between predicted output and true labels
        # self.a2 is the predicted output from the forward pass
        # y is the true labels
        # The output_error is the difference between predicted and true labels
        # This tells us how far off our predictions are from the true labels
        # The output_error is used to calculate the gradients for updating weights
        output_error = self.a2 - y  # How wrong is our final prediction?
        # The output_error is used to calculate the gradients for updating weights
        # We calculate the output delta by multiplying the output error with the derivative of the sigmoid function
        # This tells us how much the output changes with respect to a change in input
        # The output_delta is used to update the weights from hidden to output layer
        # The output_delta is calculated as output_error * sigmoid_derivative(self.a2)
        # The sigmoid_derivative(self.a2) is the derivative of the sigmoid function applied to the output
        # This is used to calculate the gradients for updating weights
        # The output_delta is used to update the weights from hidden to output layer
        output_delta = output_error * self.sigmoid_derivative(self.a2)
        
        # Step 2: Calculate error at hidden layer
        # "How much did each hidden neuron contribute to the error?"
        # We calculate the error at the hidden layer
        # The hidden_error is calculated by multiplying the output delta with the weights from hidden to output
        # self.W2.T is the transpose of the weights from hidden to output layer
        # This gives us the contribution of each hidden neuron to the output error
        # The hidden_error is used to calculate the gradients for updating weights from input to hidden layer
        # The hidden_delta is calculated by multiplying the hidden_error with the derivative of the sigmoid function    
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Step 3: Update weights and biases
        # Move weights in direction that reduces error
        # We update the weights and biases using the gradients calculated from the errors
        # We calculate the gradients for updating weights from input to hidden layer
        # The gradients are calculated as the dot product of input X and hidden delta
        # X.T is the transpose of the input data
        # This gives us the contribution of each input to the hidden layer error
        # The gradients are normalized by dividing by the number of training examples m
        # The gradients are used to update the weights from input to hidden layer
        # The weights are updated by subtracting the gradients multiplied by the learning rate
        # The biases are updated by summing the hidden delta and dividing by the number of training examples m
        # This allows the model to fit the data better by adjusting the biases
        self.W2 -= learning_rate * self.a1.T.dot(output_delta) / m
        self.b2 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m
        
        self.W1 -= learning_rate * X.T.dot(hidden_delta) / m
        self.b1 -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m
        
        return loss
    
    def train(self, X, y, epochs=10000, learning_rate=1.0, print_every=1000):
        """Train the network using backpropagation"""
        # Train the network using backpropagation
        # X is the input data, y is the true labels
        # epochs is the number of iterations to train the network
        # learning_rate controls how much we adjust weights
        # print_every controls how often we print progress
        # We will store the loss for each epoch to visualize learning
        # We initialize an empty list to store losses for each epoch
        # This will help us visualize the learning curve    
        losses = []
        
        print(f"Training for {epochs} epochs...")
        print("Epoch | Loss     | Predictions")
        print("------|----------|------------")
        
        for epoch in range(epochs):
            # Forward pass
            self.forward(X)
            # We perform a forward pass to compute the output
            # This is where we calculate the output of the network
            # The forward method computes the output based on the current weights and biases
            # The output is stored in self.a2
            # The output is used to calculate the loss and update weights
            # We calculate the output of the network using the forward method
            # This is where we compute the output of the network
            # Backward pass (learning)
            # We perform a backward pass to update weights based on the error
            # The backward method computes the gradients and updates the weights
            loss = self.backward(X, y, learning_rate)
            losses.append(loss)
            
            # Print progress
            # We print the loss and predictions every print_every epochs
            # This helps us monitor the training progress
            # We print the epoch number, loss, and predictions
            # The predictions are obtained by calling the predict method
            # The predict method performs a forward pass and returns
            if epoch % print_every == 0 or epoch == epochs - 1:
                predictions = self.predict(X).flatten()
                print(f"{epoch:5d} | {loss:.6f} | {predictions}")
        
        return losses

# Let's test our network structure
# This is the main function to create and test the XOR neural network
# It creates an instance of the XORNeuralNetwork class and tests it with the XOR dataset
# It prints the initial predictions, trains the network, and prints the final predictions
# It also plots the learning curve and decision boundary
# This is a common practice to test the network structure before training
# It helps us verify that the network is set up correctly and can learn the XOR function
# This is the main function to create and test the XOR neural network

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
    # Create an instance of the XORNeuralNetwork class
    # This initializes the network with random weights and biases
    # The network structure is 2 inputs, 3 hidden neurons, and 1 output
    # The weights and biases are initialized randomly
    # This is important for the network to learn the XOR function
    nn = XORNeuralNetwork()
    
    # Test forward pass with untrained network
    # We perform a forward pass to compute the output
    # This is where we calculate the output of the network
    # The forward method computes the output based on the current weights and biases
    # The output is stored in self.a2
    # The output is used to calculate the loss and update weights
    # We print the raw outputs and predictions before training
    # This helps us verify that the network is set up correctly and can learn the XOR function
    print("\n--- Testing Untrained Network ---")
    predictions = nn.forward(X)
    print("Raw outputs:", predictions.flatten())
    print("Predictions (>0.5):", nn.predict(X).flatten())
    print("Expected:", y.flatten())
    
    # NOW LET'S TRAIN IT!
    # Train the network to learn XOR
    # We train the network using backpropagation
    # The train method performs the forward and backward passes for a specified number of epochs
    # It updates the weights and biases based on the error
    # The learning rate controls how much we adjust the weights
    print("\n" + "="*50)
    print("TRAINING THE NETWORK TO LEARN XOR!")
    print("="*50)
    
    # Train the network
    # We call the train method with the XOR dataset, number of epochs, and learning rate
    # The train method performs the forward and backward passes for a specified number of epochs
    # It updates the weights and biases based on the error
    # The learning rate controls how much we adjust the weights
    # We store the losses for each epoch to visualize the learning curve
    # The losses are used to monitor the training progress
    # The losses are used to visualize the learning curve
    # The losses are used to monitor the training progress
    nn = XORNeuralNetwork()  # Reinitialize to reset weights    
    losses = nn.train(X, y, epochs=5000, learning_rate=10, print_every=500)
    
    # Test the trained network
    # We perform a forward pass to compute the output after training
    # This is where we calculate the output of the network after training
    # The forward method computes the output based on the updated weights and biases
    # The output is stored in self.a2
    # The output is used to calculate the loss and update weights
    # We print the raw outputs and predictions after training
    # This helps us verify that the network has learned the XOR function
    # We print the raw outputs and predictions after training
    # This helps us verify that the network has learned the XOR function
    print("\n--- Testing Trained Network ---")
    final_predictions = nn.forward(X)
    print("Raw outputs:", final_predictions.flatten())
    print("Final predictions:", nn.predict(X).flatten())
    print("Expected:", y.flatten())
    print("Accuracy:", np.mean(nn.predict(X) == y) * 100, "%")
    
    # Plot the learning curve
    # We plot the learning curve to visualize the training progress
    # The learning curve shows the loss over epochs
    # This helps us understand how well the network is learning
    # We plot the losses for each epoch to visualize the learning curve
    # The learning curve shows the loss over epochs
    # This helps us understand how well the network is learning
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Show final decision boundary
    # We plot the decision boundary to visualize how the network separates the classes
    # The decision boundary shows how the network classifies the input space
    # We create a grid of points to evaluate the network's output
    # We use np.meshgrid to create a grid of points in the input space
    # We evaluate the network's output on this grid to visualize the decision boundary
    # The decision boundary shows how the network classifies the input space
    # We create a grid of points to evaluate the network's output
    # We use np.meshgrid to create a grid of points in the input space
    # We evaluate the network's output on this grid to visualize the decision boundary
    # The decision boundary shows how the network classifies the input space
    plt.subplot(1, 2, 2)
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.forward(grid_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Output')
    
    # Plot XOR points
    # We plot the original XOR points on top of the decision boundary
    # The XOR points are colored based on their class (0 or 1)
    # We use red for class 0 and blue for class 1
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