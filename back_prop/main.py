import pandas as pd
import numpy as np
import random

class nn:
    def __init__(self, df):
        self.indexes = df.iloc[:, 0]
        self.df = df
        self.df = df.drop("label", axis=1)
        self.hidden_out = [0 for _ in range(80)]
        self.out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def pr_head(self):
        print(self.df.head())

    def activation(self, value):
        return 1/(1 + np.exp(-value))

    def expected(self, num):
        expected = []
        for i in range(len(self.out)):
            if i == num:
                expected.append(1.0)
            else:
                expected.append(0.0)
        return expected

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def train(self):
        epochs = 100
        step = 0.001
        input_bias = [1 for _ in range(80)]
        bias = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        inputs = self.df.to_numpy()
        #
        # weights = []
        # hidden_weights = []
        # for i in range(len(inputs)):
        #     input_layer_weights = []
        #     hidden_layer_weights = []
        #     for k in range(len(self.hidden_out)):
        #         input_layer_weights.append(np.array([random.random() for i in range(len(self.df.iloc[0]))]))
        #     for k in range(len(self.out)):
        #         hidden_layer_weights.append(np.array([random.random() for i in range(len(self.hidden_out))]))
        #     hidden_weights.append(hidden_layer_weights)
        #     weights.append(input_layer_weights)
        # np.save("input_weights", weights)
        # np.save("hidden_weights", hidden_weights)

        # bias = np.load("bias_weights.npy")
        # hidden_bias = np.load("hidden_b_w.npy")
        weights = np.load("input_weights.npy")
        hidden_weights = np.load("hidden_weights.npy")
        print("weights loaded!")
        print("Train len:", len(weights))
        print("Hidden layer len:", len(hidden_weights))
        print("")
        print("Output layer len:", len(self.out))
        print("num of weights for one hidden:", len(hidden_weights[0][0]))
        print("len of x", len(inputs[0]))
        print("====================")
        for epoch in range(epochs):
            print("epoch:", epoch)
            accuracy = []
            # Input layer
            for i in range(len(inputs)):
                before_softmax = []
                for k in range(len(self.hidden_out)):
                    net = np.dot(inputs[i], weights[i][k]) + input_bias[k]
                    before_softmax.append(net)
                self.hidden_out = self.softmax(before_softmax)
                # Output layer
                expected = self.expected(self.indexes[i])
                errors = []
                before_softmax = []
                for k in range(len(self.out)):
                    net = np.dot(self.hidden_out, hidden_weights[i][k]) + bias[k]
                    before_softmax.append(net)
                self.out = self.softmax(before_softmax)
                # print(self.out)
                for k in range(len(self.out)):
                    e = self.out[k] - expected[k]
                    errors.append(e)
                    out_sigma = e * self.out[k] * (1 - self.out[k])
                    bias[k] += out_sigma * step * self.out[k]
                    sigmas80 = []
                    for w_num in range(len(hidden_weights[i][k])):
                        hidden_weights[i][k][w_num] -= step * out_sigma * self.hidden_out[w_num]
                        sigmas80.append(out_sigma * hidden_weights[i][k][w_num] * (self.hidden_out[w_num]*
                                                                                   (1 - self.hidden_out[w_num])))

                for k in range(len(self.hidden_out)):
                    for w_num in range(len(weights[i][k])):
                        weights[i][k][w_num] -= step * sigmas80[k] * inputs[i][w_num]
                accuracy.append(sum(errors) / len(errors))
            np.save("bias_weights", bias)
            np.save("hidden_weights", hidden_weights)
            print("mean error:", sum(accuracy) / len(accuracy))
            np.save("input_weights", weights)



df = pd.read_csv("5k.csv")

nn1 = nn(df)
nn1.train()