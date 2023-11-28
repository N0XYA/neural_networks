import numpy as np
import random
import pandas as pd


def relu(x):
    if x <= 0:
        return 0
    else:
        return x

def dRElu(x):
    if x <= 0:
        return 0
    else:
        return 1


def f(x):
    return 2 / (1 + np.exp(-x)) - 1


def df(x):
    return 0.5 * (1 + x) * (1 - x)


hl_size = 100
w1 = np.array([[random.uniform(-0.5, 0.5) for _ in range(784)] for _ in range(hl_size)])
w2 = np.array([[random.uniform(-0.5, 0.5) for _ in range(hl_size)] for _ in range(10)])
bias1 = np.array([random.uniform(-0.5, 0.5) for _ in range(hl_size)])
bias2 = np.array([random.uniform(-0.5, 0.5) for _ in range(10)])


# w1 = np.load("w1.npy")
# w2 = np.load("w2.npy")
# bias1 = np.load("bias1.npy")
# bias2 = np.load("bias2.npy")

def go_forward(inp):
    summ = np.dot(w1, inp)
    out = np.array([f(x) for x in summ])

    summ = np.dot(w2, out)
    y = np.array([f(x) for x in summ])
    # y = f(summ)
    return (y, out)


def train(epoch):
    global w2, w1, bias2, bias1, hl_size
    step = 0.001
    right_guesses = 0
    accuracy = 0
    for i in range(len(epoch)):
        # print(i)
        expected = [1 if epoch[i][0] == _ else 0 for _ in range(10)]
        x = epoch[i]
        y, out = go_forward(x[1:])
        e = y - expected
        # e2 = sum(e[k]**2 for k in len(e))
        # e2 = 0
        # for k in e:
        #     e2 += k**2
        # e2 = 0.5 * e2
        delta = np.array([e[i] * df(y[i]) for i in range(10)])
        sigma = [0 for _ in range(hl_size)]
        # db2 =  1 / sum(delta)
        # db2 = sum(delta) / len(delta)
        # db2 = e2 / len(e)
        for n in range(10):
            w2[n] = w2[n] - step * delta[n] * y[n]
            # bias2[n] = bias2[n] + step * db2 * y[n]
            # bias2[n] = bias2[n] + step * delta[n] * bias2[n]
            for k in range(hl_size):
                sigma[k] += w2[n][k] * delta[n]

        #dont forget to delete if it doesnt work
        delta2 = [sigma[k] * df(out[k]) for k in range(hl_size)]
        # db1 = 1 / sum(delta2)
        # db1 = sum(delta2) / len(delta2)
        for n in range(hl_size):
            w1[n] = w1[n] - np.array(x[1:]) * sigma[n] * step
            # bias1[n] = bias1[n] + step * db1 * out[n]
            # bias1[n] = bias1[n] + step * sigma[n] * bias1[n]
        y = list(y)
        max_v = max(y)
        if epoch[i][0] == y.index(max_v):
            right_guesses += 1

        if right_guesses == 0 or i == 0:
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
            np.save("bias1", bias1)
            np.save("bias2.npy", bias2)
            print("weights saved!")
            print("======")

    return


set_of_numbers = pd.read_csv("../mnist_train.csv").values.tolist()
train(set_of_numbers)

