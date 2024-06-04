from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

# load dataframe
    data = load_diabetes()
    dataset_description = load_diabetes()["DESCR"]
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    
# plot data
    # extract data
    X = data.data[:, 2]     # bmi
    y = data.target         # disease progression one year after baseline

    # input data to the plot
    plt.scatter(x=X, y=y, c="blue", alpha=0.8, label="Data point", edgecolors='k', s=30)

    # add linear regression line
    coefficients = np.polyfit(X, y, 1)      # find the minimun coefficients
    trend_line = np.poly1d(coefficients)    # create trend line (a linear function) based on given coefficients
    plt.plot(X, trend_line(X), color='red', linestyle='-', label='Trend line')     # plot the trend line

    # customize the plot
    plt.title("Relationship between BMI and Diabetes Progression")
    plt.xlabel("BMI")
    plt.ylabel("Diabetes Progression")
    plt.legend(fontsize=10)
    plt.grid(True)  # add grid for better visualization

    # show the plot
    plt.show()
    