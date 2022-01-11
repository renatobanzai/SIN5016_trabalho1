import logging
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
import h5py
import datetime
import cupy as cp
import pickle
import random

class mlp_gpu:
    def __init__(self, config={}):
        logging.info("instanciado um svm")
        self.learning_rate = config["learning_rate"]
        self.input_layer_size = config["input_layer_size"]
        self.hidden_layer_size = config["hidden_layer_size"]
        self.output_layer_size = config["output_layer_size"]
        self.n_iterations = config["n_iterations"]
        self.min_cost = config["min_cost"]
        self.l2 = config["l2"]
        self.layers_size = [self.input_layer_size, self.hidden_layer_size, self.output_layer_size]
        self.parameters = {}
        self.parameters_numpy = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
        self.config = config

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

    def dictionary_to_vector(self, params_dict):
        count = 0
        for key in params_dict.keys():
            new_vector = cp.reshape(params_dict[key], (-1, 1))
            if count == 0:
                theta_vector = new_vector
            else:
                theta_vector = cp.concatenate((theta_vector, new_vector))
            count += 1

        return theta_vector

    def backward(self, X, Y, store):

        derivatives = {}

        store["A0"] = X.T

        A = store["A" + str(self.L)]
        dZ = A - Y.T

        # segundo termo para regularizacao l2
        dW = (dZ.dot(store["A" + str(self.L - 1)].T) / self.n) + self.l2/self.n * store["W" + str(self.L)]
        db = cp.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            # adicionando regulizacao l2
            dW = (1. / self.n * dZ.dot(store["A" + str(l - 1)].T)) + self.l2/self.n * store["W" + str(l)]
            db = 1. / self.n * cp.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives

    def fit(self, X, Y):
        logging.info("svm.fit")
        logging.info("config: {}".format(self.config))
        cp.random.seed(1)

        self.n = X.shape[0]

        self.layers_size.insert(0, X.shape[1])

        self.initialize_parameters()
        start_loop = datetime.datetime.now()
        for loop in range(self.n_iterations):
            A, store = self.forward(X)
            # adicionando regularizacao l2

            L2_reg= (self.l2 / (2 * self.n)) * np.sum(np.square(self.dictionary_to_vector(self.parameters)))
            cost = -cp.mean(Y * cp.log(A.T + 1e-8)) + L2_reg

            if cost <= self.min_cost:
                end_loop = datetime.datetime.now()
                logging.info("atingiu custo minimo: {}, iteracao:{}, tempo:{}".format(cost, loop, (end_loop - start_loop)))
                self.costs.append(cost)
                break

            derivatives = self.backward(X, Y, store)

            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.learning_rate * derivatives[
                    "db" + str(l)]

            if loop % 1000 == 0:
                end_loop = datetime.datetime.now()
                logging.info("Custo: {}, iteracao:{}, tempo:{}".format(cost, loop, (end_loop-start_loop)))
                start_loop = datetime.datetime.now()

            self.costs.append(cost)

        #convert cupy to numpy to enable cpu
        for x in self.parameters.keys():
            self.parameters_numpy[x] = cp.asnumpy(self.parameters[x])

        logging.info("Custo minimo: {}".format(float(min(self.costs))))

    def predict_cpu(self, X, Y):
        A, cache = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy

    def predict(self, X):
        y_softmax, cache = self.forward(X)
        y_argmax = cp.argmax(y_softmax, axis=0)
        return y_softmax, y_argmax

    def plot_cost(self):
        plt.figure()
        plt.plot(cp.asnumpy(cp.arange(len(self.costs))), [cp.asnumpy(x) for x in self.costs])
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()


    def acuracia(self, y_softmax, y):
        y_predito = cp.argmax(y_softmax, axis=0)
        y_desejado = cp.argmax(y, axis=1)
        return (y_predito==y_desejado).astype(int).mean()
