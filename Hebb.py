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
    y = [row[2] for row in df]
    w1, w2 = 0, 0
    t = 0
    k = 0
    for k in range(4):
        w1 = w1 + x1[k] * y[k]
        w2 = w2 + x2[k] * y[k]
        t = t - y[k]
        print("w1:", w1)
        print("w2:", w2)
        print("t:", t)
        print("NET:", OUT(x1[k], x2[k], w1, w2, t))
        print("=============")

    print("Введите данные для проверки:")
    print("x1:")
    test_x1 = input()
    print("x2:")
    test_x2 = input()
    print("y:", OUT(int(test_x1), int(test_x2), w1, w2, t))
    return

def main():
    training()

    return 0


if __name__ == "__main__":
    main()