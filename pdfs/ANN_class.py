import numpy as np
from sklearn import datasets
import pylab as pl


class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    nn_h_dim = 12	# hidden layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.001  # regularization strength


def generate_data():
	np.random.seed(0)
	X, y = datasets.make_moons(200, noise=0.2)
	#X, y = datasets.make_s_curve(n_samples=100,noise=0.0,random_state=0)
	
	return X, y


def visualize(X, y, model,i):
    f3 = pl.figure(3)
    pl.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=pl.cm.Spectral)
    plot_decision_boundary(lambda x:predict(model,x), X, y)
    pl.savefig("frame"+str(i/10)+".png")
    pl.pause(0.005) 

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    pl.contourf(xx, yy, Z, cmap=pl.cm.Spectral)
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.Spectral)
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.title('Clustering')
    pl.pause(0.05)
    #pl.show()


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    num_examples = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, num_passes=500, print_loss=True):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(Config.nn_input_dim, Config.nn_h_dim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, Config.nn_h_dim))
    W2 = np.random.randn(Config.nn_h_dim, Config.nn_output_dim) / np.sqrt(Config.nn_h_dim)
    b2 = np.zeros((1, Config.nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1

        # Gradient descent parameter update
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        
        if print_loss and i % 10 == 0:
            f1  = pl.figure(1)
            pl.plot(i,calculate_loss(model, X, y),'bo')
            visualize(X, y, model,i)
            pl.pause(0.001)

    return model

def main():
    X, y = generate_data()
    model = build_model(X, y, print_loss=True)
    f4 = pl.figure(4)
    pl.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=pl.cm.Spectral)
    plot_decision_boundary(lambda x:predict(model,x), X, y)
    pl.show()

if __name__ == "__main__":
    main()
