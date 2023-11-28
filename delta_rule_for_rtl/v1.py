from sklearn.datasets import load_digits
import pylab as pl
import random


digits = load_digits()
target = digits.target.tolist()
digits = digits.data.tolist()
train_d = digits[:1500]
train_t = target[:1500]
test_d = digits[1500:]
test_t = target[1500:]
hl_size = 10
w = [[random.randint(0, 1) for _ in range(64)] for _ in range(10)]
w1 = [[random.randint(0,1) for _ in range(64)] for _ in range(hl_size)]
w2 = [[random.randint(0, 1) for _ in range(hl_size)] for _ in range(10)]

bias1 = [[random.randint(0,1) for _ in range(64)] for _ in range(hl_size)]
bias2 = [[random.randint(0, 1) for _ in range(hl_size)] for _ in range(10)]


def f(x):
    return 1 if x>= 1 else 0


def make_target_dict(target):
    out = [1 if _ == target else 0 for _ in range(10)]
    return out


def go_forward(input):
    out = []
    y = []
    for i in range(hl_size):
        summ = 0
        for k in range(len(input)):
            summ += input[k] * w1[i][k]
            # print(k)
        out.append(f(summ))
    # print(out)

    for i in range(10):
        summ = 0
        for k in range(hl_size):
            summ += out[k] * w2[i][k]
        y.append(f(summ))
    return out, y


def go_forward_1l(input):
    y = []
    for i in range(10):
        summ = 0
        for k in range(len(input)):
            summ += input[k] * w[i][k]
        y.append(f(summ))
    return y

def weight_adj(x, err):
    for i in range(10):
        for k in range(len(x)):
            w[i][k] = w[i][k] + err[i] * x[k]

def train(digits, target):
    for i in range(len(digits)):
        x = digits[i]
        expected = make_target_dict(target[i])
        print(target[i])
        print("expected", expected)
        # out, y = go_forward(x)
        y = go_forward_1l(x)
        print("got     ", y)
        errors = [0 for _ in range(10)]
        for ind in range(10):
            errors[ind] = expected[ind] - y[ind]
        weight_adj(x, errors)
        print(i)
        print("======")


def test(digits, target):
    right_guesses = 0
    accuracy = 0
    for i in range(len(digits)):
        x = digits[i]
        expected = make_target_dict(target[i])
        print(target[i])
        print("expected", expected)
        y = go_forward_1l(x)
        print("got     ", y)
        if expected == y:
            right_guesses += 1
        if right_guesses == 0 or i == 0:
            accuracy == 0.01
        else:
            accuracy = right_guesses / i
        print(f"Accuracy = {accuracy * 100} %")
        print("===")

    print(f"overall accuracy: {accuracy * 100} %")


train(train_d, train_t)
test(test_d, test_t)
