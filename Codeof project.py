import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
print("Now I loaded the healthy haert data in csv formmat")
heart_data = pd.read_csv('/content/hds.csv.csv')
print("Now i used the Head to print the row of the csv file")
heart_data.head()
print("Now I used the tail to print the last rows of the data")
heart_data.tail()
print("number of rows and columns in the dataset")
heart_data.shape
print("getting some info about the data")
heart_data.info()
print("checking for missing values")
heart_data.isnull().sum()
print("for the statistical measures about the data")
heart_data.describe()
print("checking the distribution of Target Variable")
heart_data['target'].value_counts()
print("Now I divided the data into the X and the Y axis: "")
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)
print("This I does to make the data to flow in a continuous state and moreover the data can be read into a linear manner for the easy tackle")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print("Now from here the logistic regression starts and it is used for the probability of the heart disease ")
model = LogisticRegression()
print("Below I find out the accuracy on training data or the csv file")
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
print("accuracy on test data, the data that we input for the test")
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)

print("Now here the input is given")
print("The below data include the following factors as Age, sex(1=Male and 0=Female, constrictive precarditis, resting bps, cholestrol level, fasting glucose level, maxximum bps,resting EGC,
and other factors too")
input_data = (62,2,5,6,5,2,5,2,6,4,5,8,9)

print("changing the csv file to the array using numpy")
input_data_as_numpy_array= np.asarray(input_data)

print(" reshape the numpy array as we are predicting for only on instance")
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
