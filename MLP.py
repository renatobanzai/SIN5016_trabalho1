import numpy as np
import math

class MLP:
    def __init__(self,
                 input_layer_length=0,
                 hidden_layer_length=0,
                 output_layer_length=1,
                 h_activation_function="sigmoid",
                 o_activation_function="sigmoid",
                 debug=True):

        self.input_layer_length = input_layer_length
        self.hidden_layer_length = hidden_layer_length
        self.output_layer_length = output_layer_length

        if h_activation_function.lower() == "sigmoid":
            self.h_activation_function = self.sigmoid
            self.dh_activation_function = self.d_sigmoid
        if o_activation_function.lower() == "sigmoid":
            self.o_activation_function = self.sigmoid
            self.do_activation_function = self.d_sigmoid

        if h_activation_function.lower() == "softmax":
            self.h_activation_function = self.softmax
            self.dh_activation_function = self.d_softmax
        if o_activation_function.lower() == "softmax":
            self.o_activation_function = self.softmax
            self.do_activation_function = self.d_softmax

        self.w_hidden_layer = np.random.randint(
            low=-5000,
            high=5001,
            size=(self.hidden_layer_length,
                  self.input_layer_length + 1)) / 10000

        self.w_output_layer = np.random.randint(
            low=-5000,
            high=5001,
            size=(self.output_layer_length,
                  self.hidden_layer_length + 1)) / 10000

        # self.w_output_layer = np.random.rand(self.output_layer_length,
        #                                      self.hidden_layer_length + 1)

        self.debug = debug

        if self.debug: print(self.w_hidden_layer, self.w_output_layer)

    def sigmoid(self, net):
        return (1 / (1 + np.exp(-net)))

    def d_sigmoid(self, f_net):
        return (f_net * (1 - f_net))

    def softmax(self, net):
        return 0  # todo: implementar

    def d_softmax(self, f_net):
        return 0  # todo: implementar

    def forward(self, Xp):
        net_hidden_layer = self.w_hidden_layer.dot(np.c_[Xp, 1].T)
        f_net_hidden_layer = self.h_activation_function(net_hidden_layer)

        net_output_layer = self.w_output_layer.dot(np.c_[f_net_hidden_layer.T,1].T)
        f_net_output_layer = self.o_activation_function(net_output_layer)

        if self.debug: print(net_hidden_layer, net_output_layer, f_net_hidden_layer, f_net_output_layer)
        return net_hidden_layer, net_output_layer, f_net_hidden_layer, f_net_output_layer

    def back_propagation(self, X, Y, eta=0.1, threshold=1e-3, max_iter=50000):
        iter = 0
        sqrError = 2 * threshold
        allSqrError = []
        while (sqrError > threshold and iter < max_iter):
            iter += 1
            sqrError = 0
            for p in range(X.shape[0]):
                Xp = X[p]
                Xp.shape = (1, X.shape[1])
                net_hidden_layer, net_output_layer, f_net_hidden_layer, f_net_output_layer = self.forward(Xp)
                error = Y[p] - f_net_output_layer
                sqrError += np.sum(np.power(error, 2))

                delta_o_p = error * self.do_activation_function(f_net_output_layer)

                w_o_kj = self.w_output_layer[:, 0:self.hidden_layer_length]
                # w_o_kj = self.w_output_layer[:, :]

                delta_h_p = self.dh_activation_function(f_net_hidden_layer).T * (delta_o_p.dot(w_o_kj))

                # training
                self.w_output_layer += eta * (delta_o_p.dot(np.c_[f_net_hidden_layer.T, 1]))

                self.w_hidden_layer += eta * (delta_h_p.T.dot(np.c_[Xp, 1]))

            sqrError = sqrError / X.shape[0]
            allSqrError.append(sqrError)
            if self.debug: print(f"Erro: {sqrError}")

        print("AllError:{}".format(allSqrError))
        print("Épocas:{}".format(iter))
        print("sqrError:{}".format(sqrError))

    def fit(self):
        return

    def predict(self):
        return


if __name__=="__main__":
    mlp = MLP(2,3,1, debug=False)

    test = np.c_[0, 0]
    print(mlp.forward(test))

    X = np.array([[0,0],[0,1], [1,0], [1,1]])
    Y = np.array([0,1,1,0])

    mlp.back_propagation(X, Y)

    test = np.c_[0, 0]
    print(mlp.forward(test))

    test = np.c_[1, 0]
    print(mlp.forward(test))

    test = np.c_[0, 1]
    print(mlp.forward(test))

    test = np.c_[1, 1]
    print(mlp.forward(test))

