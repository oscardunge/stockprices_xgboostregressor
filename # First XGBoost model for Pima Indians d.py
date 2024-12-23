# First XGBoost model for Pima Indians dataset
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
# import numpy as np

# dataset = np.loadtxt('C:\Users\oscar\Downloads\pima-indians-diabetes.csv', delimiter=",", encoding='utf-8')





import numpy as np

dataset = np.loadtxt(r'C:\Users\oscar\Downloads\pima-indians-diabetes.csv', delimiter=",")


print(dataset)

print(dataset[0:,0:8])

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

print(Y)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))









