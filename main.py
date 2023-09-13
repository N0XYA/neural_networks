import pandas as pd

k = 1 #iterations

df = pd.read_excel('training_data.xlsx')


def OUT(NET):
    if NET > 0:
        return 1
    else:
        return -1


def training(df):
    T = 0
    k = 4
    w1 = 0
    w2 = 0
    x1_values = [value[0] for value in df.values]
    x2_values = [value[1] for value in df.values]
    # y_values = [value[2] for value in df.values]
    for i in range(k):
        y = OUT(x)
        x1 = x1_values[i]
        x2 = x2_values[i]
        w1 = w1 + x1 * y
        w2 = w2 + x2 * y
        T = T - y
        NET = 1 if x1 == x2 else 0
        # NET = OUT(NET)
        print("i=", i, "x1=", x1, "x2=", x2, "y=", y)
        print("w1 =", w1, "w2=", w2, "T=", T, "NET=", NET)
        print('\n')
    return w1, w2


def main():
    w1, w2 =training(df)
    print(w1, w2)
    return 0


if __name__ == "__main__":
    main()