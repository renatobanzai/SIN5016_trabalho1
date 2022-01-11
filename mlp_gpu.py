import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
import h5py
import datetime
import cupy as cp
import pickle
import random

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
            # inicializacao como no artigo sugerido
            r = np.sqrt(1 / self.layers_size[l-1])
            self.parameters["W" + str(l)] = cp.random.uniform(-r, r, (self.layers_size[l], self.layers_size[l - 1]))
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
        print("learning rate: " + str(learning_rate))
        print("epochs: " + str(n_iterations))

        cp.random.seed(1)

        self.n = X.shape[0]

        self.layers_size.insert(0, X.shape[1])

        self.initialize_parameters()
        start_loop = datetime.datetime.now()
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
                end_loop = datetime.datetime.now()
                print("Cost: ", cost, "Train Accuracy:", self.predict(X, Y), "x loop time", (end_loop-start_loop))
                start_loop = datetime.datetime.now()

            self.costs.append(cost)

        #convert cupy to numpy to enable cpu
        for x in self.parameters.keys():
            self.parameters_numpy[x] = cp.asnumpy(self.parameters[x])

    def predict_cpu(self, X, Y):
        A, cache = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100

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


def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = np.array(train_y.get())
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
    train_y = cp.array(train_y)

    test_y = np.array(test_y.get())
    test_y = enc.transform(test_y.reshape(len(test_y), -1))
    test_y = cp.array(test_y)

    return train_x, train_y, test_x, test_y


def pre_process_data_hog(train_x, train_y, test_x, test_y):
    # Normalize
    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = np.array(train_y.get())
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
    train_y = cp.array(train_y)

    test_y = np.array(test_y.get())
    test_y = enc.transform(test_y.reshape(len(test_y), -1))
    test_y = cp.array(test_y)

    return train_x, train_y, test_x, test_y

def split_data(dataset, id, train_percent):
  dataset = dataset[cp.where(dataset[:,0]==id)]
  train_registers = int(dataset.shape[0] * train_percent)
  test_registers = dataset.shape[0] - train_registers
  if test_registers == 0 or train_registers == 0:
    print("Id {} com registros insuficientes.".format(id))
    x_train = dataset[0:0, 1:]
    y_train = dataset[0:0, [0]]
    x_test = dataset[0:0, 1:]
    y_test = dataset[0:0, [0]]
  else:
    x_train = dataset[0:train_registers, 1:]
    y_train = dataset[0:train_registers, [0]]
    x_test = dataset[test_registers:, 1:]
    y_test = dataset[test_registers:, [0]]

  return x_train, y_train, x_test, y_test


def train_hog(path_hog):
    time_start = datetime.datetime.now()
    f = h5py.File(path_hog, 'r')

    hog_array = cp.array(f['descriptor'])  # numpy array


    qtd_classes = 1000
    percent_train = 0.6



    random.seed(2)
    classes = np.array(cp.unique(hog_array[:, 0]).get())
    random.shuffle(classes)
    classes = classes[0: qtd_classes]
    classes = cp.array(classes)
    # inicializa com o id 0
    x_train_hog, y_train_hog, x_test_hog, y_test_hog = split_data(hog_array, 0, percent_train)

    for classe in classes:
        # concatena os conjuntos
        x_tr, y_tr, x_te, y_te = split_data(hog_array, classe, percent_train)
        x_train_hog = cp.concatenate([x_train_hog, x_tr])
        y_train_hog = cp.concatenate([y_train_hog, y_tr])
        x_test_hog = cp.concatenate([x_test_hog, x_te])
        y_test_hog = cp.concatenate([y_test_hog, y_te])

    train_x, train_y, test_x, test_y = pre_process_data_hog(x_train_hog, y_train_hog, x_test_hog, y_test_hog)

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    layers_dims = [80, cp.unique(y_train_hog[:, 0]).shape[0]]

    ann = ANN(layers_dims)
    n_iterations = 10000
    learning_rate = 0.25
    ann.fit(train_x, train_y, learning_rate=learning_rate, n_iterations=n_iterations)

    time_end = datetime.datetime.now()
    print("Train Time:", (time_end-time_start))
    print("Train Accuracy:", ann.predict(train_x, train_y))
    print("Test Accuracy:", ann.predict(test_x, test_y))

    pickle.dump(ann, open("mlp_hog_{}_{}_{}.dat".format(qtd_classes,n_iterations, learning_rate), "wb"))

    ann.plot_cost()

def train_lbp(path_lbp):
    time_start = datetime.datetime.now()
    f = h5py.File(path_lbp, 'r')

    hog_array = cp.array(f['descriptor'])  # numpy array


    qtd_classes = 200
    percent_train = 0.6



    random.seed(2)
    classes = np.array(cp.unique(hog_array[:, 0]).get())
    random.shuffle(classes)
    classes = classes[0: qtd_classes]
    classes = cp.array(classes)
    # inicializa com o id 0
    x_train_hog, y_train_hog, x_test_hog, y_test_hog = split_data(hog_array, 0, percent_train)

    for classe in classes:
        # concatena os conjuntos
        x_tr, y_tr, x_te, y_te = split_data(hog_array, classe, percent_train)
        x_train_hog = cp.concatenate([x_train_hog, x_tr])
        y_train_hog = cp.concatenate([y_train_hog, y_tr])
        x_test_hog = cp.concatenate([x_test_hog, x_te])
        y_test_hog = cp.concatenate([y_test_hog, y_te])

    train_x, train_y, test_x, test_y = pre_process_data(x_train_hog, y_train_hog, x_test_hog, y_test_hog)

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    layers_dims = [150, cp.unique(y_train_hog[:, 0]).shape[0]]

    ann = ANN(layers_dims)
    n_iterations = 10000
    learning_rate = 8.75
    ann.fit(train_x, train_y, learning_rate=learning_rate, n_iterations=n_iterations)

    time_end = datetime.datetime.now()
    print("Train Time:", (time_end-time_start))
    print("Train Accuracy:", ann.predict(train_x, train_y))
    print("Test Accuracy:", ann.predict(test_x, test_y))

    pickle.dump(ann, open("mlp_lbp_{}_{}_{}.dat".format(qtd_classes,n_iterations, learning_rate), "wb"))

    ann.plot_cost()

# train_lbp("./data/lbp_11_15_18_20") #lbp errado
# train_lbp("./data/lbp_grid")
train_hog("./data/hog_11_15_20_56")


