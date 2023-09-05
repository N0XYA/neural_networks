import pandas as pd

k = 1 #iterations

df = pd.read_excel('training_data.xlsx')


w1 = 0
w2 = 0
def training(df):
    x1 = [value[0] for value in df.values]
    x2 = [value[1] for value in df.values]
    y = [value[2] for value in df.values]
    return


def main():
    training(df)
    return 0


if __name__ == "__main__":
    main()