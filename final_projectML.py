import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    
    df = pd.read_csv("Malware_subset.csv")
    encoder = LabelEncoder()
    df["Label"] = encoder.fit_transform(df["Label"])

    X = df.drop(columns="Label").values
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("The score is: {}".format(score))