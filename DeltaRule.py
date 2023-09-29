import pandas as pd

df = pd.read_excel('delta_rule.xlsx')
df = df.values.tolist()

def out(net):
    return 1 if net >= 2 else -1


def net(x1, x2, w1, w2, t, y):
    return x1 * w1 + x2 * w2 - y * t


def training():
    x1 = [row[0] for row in df]
    x2 = [row[1] for row in df]
    y = [row[2] for row in df]
    n = 1
    e = 0
    w1 = 0
    w2 = 0
    t = 0
    for i in range(len(x1)):
        print("x1=", x1[i], "x2=", x2[i], "y=", y[i])
        s = net(x1[i], x2[i], w1, w2, t, y[i])
        pred = out(s)
        print("net=", s, "y pred=", pred)
        w1 = w1 + n * (y[i] - pred) * x1[i]
        w2 = w2 + n * (y[i] - pred) * x2[i]
        t = t + n * (y[i] - pred) * t
        print("w1=", w1, "w2=", w2)
        print("t=", t)
        print("=========")

    print("TEST")
    test1 = int(input("x1:"))
    test2 = int(input("x2:"))
    print(out(test1*w1 + test2*w2))
def main():
    training()


if __name__ == '__main__':
    main()