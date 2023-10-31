import numpy as np
import pandas as pd

w1 = np.load("w1.npy")
w2 = np.load("w2.npy")
bias1 = np.load("bias1.npy")
bias2 = np.load("bias2.npy")

def f(x):
    return 2 / (1 + np.exp(-x)) - 1


def relu(x):
    if x <= 0:
        return 0
    else:
        return x


def go(inp):
    summ = np.dot(w1, inp)
    out = np.array([f(x) for x in summ])

    summ = np.dot(w2, out)
    y = np.array([f(x) for x in summ])
    # y = f(summ)
    return y


epoch = pd.read_csv("../mnist_test.csv").values.tolist()

right_guesses = 0
accuracy = 0
for i in range(len(epoch)):
    expected = [1 if epoch[i][0] == _ else 0 for _ in range(10)]
    x = epoch[i]
    y = go(x[1:])
    y = list(y)
    max_v = max(y)
    if epoch[i][0] == y.index(max_v):
        right_guesses += 1

    if right_guesses == 0 or i == 0:
        accuracy = 0.001
    else:
        accuracy = right_guesses / i
    if i % 1000 == 0:
        print("epoch ", i)
        # print(f"NUMBER {epoch[i][:1]}")
        print("Expected ", epoch[i][0])
        print("got", y.index(max_v))
        print(f"accuracy {accuracy * 100} %")
        print(right_guesses, i)
        print("==========")

print(f"overall accuracy {accuracy * 100} %")