#Simple linear regression 
#Dataset: https://www.kaggle.com/andonians/random-linear-regression

# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Load the train and test files 
train = pd.read_csv('D:/Python/input/train.csv')
test = pd.read_csv('D:/Python/input/test.csv')

#Remove rows with empty columns
train = train.dropna() 
test = test.dropna()

#Print first few rows of data
print('First 5 values of training data')
print(train.head(),'\n')
print ('First 5 values of test data')
print(test.head(),'\n')

#Convert to array format and separate the input feature and the target variable
x_train=train['x'].to_numpy().reshape(-1,1).astype(float)
y_train=train['y'].to_numpy().reshape(-1,1).astype(float)

#Scatter plot to visualize the relation between the feature and target 
plt.figure(figsize=(10, 10))
plt.scatter(
    train['x'],
    train['y'],
    c='blue'
)
plt.xlabel("X")
plt.ylabel("y")
plt.show()
#The plot suggests a strong linear relationship

#Find the value of R-square and p-value of the data
x_train2 = sm.add_constant(x_train)
obj = sm.OLS(y_train,x_train2)
res = obj.fit()
print(res.summary(),'\n')
#The value of R-square (which is close to 1) and the low p-value of the coefficient of the independent variable indicate that there is a strong correlation between the input and output variables

#Apply Linear Regression function and fit to train data
reg = LinearRegression()
reg.fit(x_train,y_train)
print("The line fit by the regression function is y = {:.4} x + {:.4}".format(reg.intercept_[0], reg.coef_[0][0]))

#Predict on the test data
x_test=test['x'].to_numpy().reshape(-1,1).astype(float)
y_test=test['y'].to_numpy().reshape(-1,1).astype(float)
predictions = reg.predict(x_test)

#Visualize the line that fits the data
plt.figure(figsize=(10, 10))
plt.scatter(
    test['x'],
    test['y'],
    c='black'
)
plt.plot(x_test,predictions,color='red')
plt.show()
