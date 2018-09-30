from model import DNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np


def fetch_data():
    X_train, y_train = [], []
    k = 1
    with open('./../Input/training_data.csv') as f:
        for line in f:
            data = list(map(float, line.strip().split(',')))
            y_train.append(data[-1])
            X_train.append(data[:-1])
    return np.array(X_train), np.array(y_train), k


# intitializing the data
X_train, y_train, k = fetch_data()
# split the data into train and test randomly
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('Starting')
# call our model
regr = DNN()
regr.fit(X_train, y_train, k)
prediction = regr.predict(X_test)
error = mean_absolute_error(y_true=y_test, y_pred=prediction)
print('Test error is: ', error)
