import copy
import numpy as np


class Rnn:
    def __init__(self, bin_size=8, input_dim=2, hidden_dim=20, output_dim=1):
        self.bin_size = bin_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.binary_list = np.unpackbits(np.array([range(2 ** self.bin_size)], dtype=np.uint8).T, axis=1)
        self.x_weights = np.random.randn(self.input_dim, self.hidden_dim)
        self.y_weights = np.random.randn(self.hidden_dim, self.output_dim)
        self.H_derivs = np.random.randn(self.hidden_dim, self.hidden_dim)
        self.x_weights_update = np.zeros_like(self.x_weights)
        self.y_weights_update = np.zeros_like(self.y_weights)
        self.H_derivs_update = np.zeros_like(self.H_derivs)

    def sigmoid(self, x, deriv=False):
        if deriv == False:
            return 1 / (1 + np.exp(-x))
        if deriv == True:
            return x * (1 - x)

    def train(self, learning_rate=0.1, epochs=100000, verbose=True):
        for j in range(epochs):

            # create values to add (not more then 7 bit large)
            # ----------------------
            a_int = np.random.randint(2 ** self.bin_size / 2)
            b_int = np.random.randint(2 ** self.bin_size / 2)
            a = self.binary_list[a_int]
            b = self.binary_list[b_int]

            # this is Y_true
            c = self.binary_list[a_int + b_int]

            # where Y_predicts will be stored
            d = np.zeros_like(c)  # only needed for tracking

            overallError = 0  # only needed for tracking (total error this epoch)

            Y_derivs = list()
            H_values = list()
            H_values.append(np.zeros(self.hidden_dim))

            # moving along the position in the binary encoding
            for position in range(self.bin_size):
                # generate input and output
                x = np.array([[a[self.bin_size - position - 1], b[self.bin_size - position - 1]]])
                y = np.array([[c[self.bin_size - position - 1]]]).T

                # hidden layer (input ~+ prev_hidden)
                Ht_value = self.sigmoid(np.dot(x, self.x_weights) + np.dot(H_values[-1], self.H_derivs), deriv=False)

                # output layer (input ~+ prev_hidden)
                Yt_value = self.sigmoid(np.dot(Ht_value, self.y_weights), deriv=False)

                # estimating loss
                Yt_value_error = Yt_value - y
                Y_derivs.append((Yt_value_error) * self.sigmoid(Yt_value, deriv=True))
                overallError += np.abs(Yt_value_error[0])

                # add to Y_predict
                d[self.bin_size - position - 1] = np.round(Yt_value[0][0])

                # store hidder layer values
                H_values.append(copy.deepcopy(Ht_value))

            # Main start of backprop
            future_Ht_deriv = np.zeros(self.hidden_dim)

            for position in range(self.bin_size):
                x = np.array([[a[position], b[position]]])
                Ht_value = H_values[-position - 1]
                prev_Ht_value = H_values[-position - 2]

                # error at output layer
                Yt_deriv = Y_derivs[-position - 1]
                # error at hidden layer
                Ht_deriv = (future_Ht_deriv.dot(self.H_derivs.T) + Yt_deriv.dot(self.y_weights.T)) * \
                           self.sigmoid(Ht_value, deriv=True)

                # update weights and go again
                self.y_weights_update += np.atleast_2d(Ht_value).T.dot(Yt_deriv)
                self.H_derivs_update += np.atleast_2d(prev_Ht_value).T.dot(Ht_deriv)
                self.x_weights_update += x.T.dot(Ht_deriv)

                future_Ht_deriv = Ht_deriv

            self.x_weights -= self.x_weights_update * learning_rate
            self.y_weights -= self.y_weights_update * learning_rate
            self.H_derivs -= self.H_derivs_update * learning_rate

            self.H_derivs_update *= 0
            self.x_weights_update *= 0
            self.y_weights_update *= 0

            if verbose == True:
                # print progress
                if (j % 1000 == 0):
                    print("Epoch ", j, "/", epochs)
                    print("Error:" + str(overallError) + "\nPred:" + str(d) + "\nTrue:" + str(c))
                    out = 0
                    for index, x in enumerate(reversed(d)):
                        out += x * pow(2, index)
                    print(str(a_int) + " + " + str(b_int) + " = " + str(out))
                    print("---------------")


attempt1 = Rnn()
attempt1.train(0.1,20000,1)
