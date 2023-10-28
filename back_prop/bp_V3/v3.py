import numpy as np
import random
import pandas as pd


def f(x):
    return 2 / (1 + np.exp(-x)) - 1


def df(x):
    return 0.5 * (1 + x) * (1 - x)


hl_size = 250
w1 = np.array([[random.uniform(-0.5, 0.5) for _ in range(784)] for _ in range(hl_size)])
w2 = np.array([[random.uniform(-0.5, 0.5) for _ in range(hl_size)] for _ in range(10)])


def go_forward(inp):
    summ = np.dot(w1, inp)
    out = np.array([f(x) for x in summ])

    summ = np.dot(w2, out)
    y = np.array([f(x) for x in summ])
    # y = f(summ)
    return (y, out)


def train(epoch):
    global w2, w1, hl_size
    step = 0.001
    right_guesses = 0
    accuracy = 0
    for i in range(len(epoch)):
        # print(i)
        expected = [1 if epoch[i][0] == _ else 0 for _ in range(10)]
        x = epoch[i]
        y, out = go_forward(x[1:])
        e = y - expected
        delta = np.array([e[i] * df(y[i]) for i in range(10)])
        sigma = [0 for _ in range(hl_size)]
        for n in range(10):
            w2[n] = w2[n] - step * delta[n] * y[n]
            for k in range(hl_size):
                sigma[k] += w2[n][k] * delta[n]

        for n in range(hl_size):
            w1[n] = w1[n] - np.array(x[1:]) * sigma[n] * step

        y = list(y)
        max_v = max(y)
        if epoch[i][0] == y.index(max_v):
            right_guesses += 1

        if right_guesses == 0:
            accuracy = 0
        else:
            accuracy = right_guesses / i
        if i % 1000 == 0:
            print("epoch ", i)
            # print(f"NUMBER {epoch[i][:1]}")
            print("Expected ", epoch[i][:1])
            print("got", y.index(max_v))
            np.save("w1", w1)
            np.save("w2", w2)
            print("weights saved!")
            print("======")

    return


set_of_numbers = pd.read_csv("../mnist_train.csv").values.tolist()
train(set_of_numbers)

