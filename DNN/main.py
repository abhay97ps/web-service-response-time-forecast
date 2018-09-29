from model import DNN
from sklearn.model_selection import train_test_split


def fetch_data():
    # to be coded or imported
    X_train, y_train, k = [], [], 0
    return X_train, y_train, k


# intitializing the data
X_train, y_train, k = fetch_data()
# split the data into train and test set not randomly
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False)
# call our model
regr = DNN()
regr.fit(X_train, y_train, k)
prediction = regr.predict(X_test)
