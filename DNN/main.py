from model import DNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def fetch_data():
    X_train, y_train = [], []
    k = 1
    with open('./../Input/training_data_10.csv') as f:
        for line in f:
            data = list(map(float, line.strip().split(',')))
            y_train.append(data[-1])
            X_train.append(data[:-1])
    return np.array(X_train), np.array(y_train), k


# intitializing the data
print('Reading Data')
X_train, y_train, k = fetch_data()
print('Done')
# split the data into train and test randomly
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2)

scaler = MinMaxScaler(feature_range=(1, 10))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train.reshape([y_train.shape[0],1]))
y_test = scaler.transform(y_test.reshape([y_test.shape[0],1]))
    
# scaling input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# making target array a column vector
y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('Starting')
print('Training the model')
# call our model
regr = DNN()
prediction = regr.fit(X_train, y_train, k, X_test)
error = np.mean(abs(1-(y_test/(prediction+1e-8))))
print('Test error is: ' + str(error*100.00) + '%')
