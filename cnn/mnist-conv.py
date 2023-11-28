import numpy as np
from sklearn.datasets import load_digits
import random

digits = load_digits()
target = digits.target
digits = digits.data
train_d = digits[:1500]
train_t = target[:1500]
test_d = digits[1500:]
test_t = target[1500:]

test_digit = train_d[0]
test_target = train_t[0]
# print(test_target)
# print(len(test_digit))


def f(x):
    return 1 if x > 0 else 0


kernel = np.array([random.uniform(-1, 1) for i in range(9)])
# print(kernel)
conv = np.convolve(test_digit, kernel, mode="valid")
for i in range(len(conv)):
    conv[i] = f(conv[i])
    # print(conv[i])
print(len(conv))

pooling = []
for i in range(0, len(conv), 2):
    buff = []
    for k in range(4):
        buff.append(conv[k])
    pooling.append(max(buff))

print(len(pooling))