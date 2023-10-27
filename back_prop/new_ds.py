import pandas as pd

df = pd.read_csv("mnist_train.csv")
new_df = df.iloc[:5000]
new_df.to_csv(r'5k.csv', index= False)