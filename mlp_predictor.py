import pickle
import numpy as np
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
import h5py
import datetime

class ANN:
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.parameters_numpy = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []

    def sigmoid_numpy(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax_numpy(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def sigmoid(self, Z):
        return 1 / (1 + cp.exp(-Z))

    def softmax(self, Z):
        expZ = cp.exp(Z - cp.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def initialize_parameters(self):
        cp.random.seed(1)

        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = cp.random.randn(self.layers_size[l], self.layers_size[l - 1]) / cp.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = cp.zeros((self.layers_size[l], 1))

    def forward(self, X):
        store = {}

        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z

        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z

        return A, store

    def forward_cpu(self, X):
        store = {}

        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters_numpy["W" + str(l + 1)].dot(A) + self.parameters_numpy["b" + str(l + 1)]
            A = self.sigmoid_numpy(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters_numpy["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z

        Z = self.parameters_numpy["W" + str(self.L)].dot(A) + self.parameters_numpy["b" + str(self.L)]
        A = self.softmax_numpy(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters_numpy["W" + str(self.L)]
        store["Z" + str(self.L)] = Z

        return A, store

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + cp.exp(-Z))
        return s * (1 - s)

    def backward(self, X, Y, store):

        derivatives = {}

        store["A0"] = X.T

        A = store["A" + str(self.L)]
        dZ = A - Y.T

        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = cp.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * cp.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives

    def fit(self, X, Y, learning_rate=0.01, n_iterations=2500):
        cp.random.seed(1)

        self.n = X.shape[0]

        self.layers_size.insert(0, X.shape[1])

        self.initialize_parameters()
        for loop in range(n_iterations):
            A, store = self.forward(X)
            cost = -cp.mean(Y * cp.log(A.T + 1e-8))
            derivatives = self.backward(X, Y, store)

            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]

            if loop % 1000 == 0:
                print("Cost: ", cost, "Train Accuracy:", self.predict(X, Y))

            self.costs.append(cost)

        #convert cupy to numpy to enable cpu
        for x in self.parameters.keys():
            self.parameters_numpy[x] = cp.asnumpy(self.parameters[x])

    def predict_cpu(self, X, Y):
        A, cache = self.forward_cpu(X)
        y_hat = np.argmax(A, axis=0)
        return y_hat

    def predict(self, X, Y):
        A, cache = self.forward(X)
        y_hat = cp.argmax(A, axis=0)
        Y = cp.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100

    def plot_cost(self):
        plt.figure()
        plt.plot(cp.asnumpy(cp.arange(len(self.costs))), [ cp.asnumpy(x) for x in self.costs])
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()


mlp = pickle.load(open("model/mlp_.p", "rb"))

time_start = datetime.datetime.now()
path_hog = "./data/hog_11_15_20_56"
f = h5py.File(path_hog, 'r')
hog_array = f['descriptor']

registro = hog_array[np.where(hog_array[:,0]==112)][3,1:].reshape(1,576)

print(mlp.predict_cpu(registro, registro))
print("")