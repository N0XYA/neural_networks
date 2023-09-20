import pandas as pd

df = pd.read_excel('training_data.xlsx')
df = df.values.tolist()


def OUT(x1, x2, w1, w2, t):
    NET = w1*x1 + w2*x2 - t
    if NET > 0:
        return 1
    else:
        return -1

def training():
    x1 = [row[0] for row in df]
    x2 = [row[1] for row in df]
    x3 = -1
    y = [row[2] for row in df]
    w1, w2 = 0, 0
    t = 0
    n = 1
    e = []
    for i in range(len(x1)):
        print(i)
        e.append(y[i] - OUT(x1[i], x2[i], w1, w2, t))
        w1 = w1 + n * (y[i] - OUT(x1[i], x2[i], w1, w2, t)) * x1[i]
        w2 = w2 + n * (y[i] - OUT(x1[i], x2[i], w1, w2, t)) * x2[i]
        t = t + n * (y[i] - OUT(x1[i], x2[i], w1, w2, t)) * x3
        print("w1", w1)
        print("w2", w2)
        print("out", OUT(x1[i], x2[i], w1, w2, t))
        print("===============")
    print(w1, w2)
    print(e)
    print("Test x1")
    testx1 = input()
    print("test x2")
    testx2 = input()
    print("y:", OUT(int(testx1), int(testx2), w1, w2, t))
def main():
    training()


if __name__ == '__main__':
    main()