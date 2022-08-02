import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("./iris.csv", header=None)
    print(df.head(5))
    # print(df.groupby(4).count())

    train, test = train_test_split(df, test_size=0.5, stratify=df[4], shuffle=True, random_state=420)
    print(train)


if __name__ == '__main__':
    main()