import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the dataset
df=pd.read_csv("FuelConsumption.csv")

#Now split the dataset into train=80% and test=20% sets using np.random.rand
msk=np.random.rand(len(df))<0.8
train=df[msk]
test=df[~msk]

#Now import LinearRegression from sklearn
from sklearn import linear_model
regr=linear_model.LinearRegression()
#Now create an array using the train data
train_x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
#printing the coefficients
print("The coefficient and intercept are:",regr.coef_,regr.intercept_)

#EVALUATION OF THE MODEL with the test set
from sklearn.metrics import r2_score
test_x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
y_pred=regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred- test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , y_pred) )