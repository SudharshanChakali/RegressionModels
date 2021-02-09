import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the dataset
df=pd.read_csv("FuelConsumption.csv")
print(df.describe()) #gives description of the dataset

#let's see how ENGINESIZE AND CO2EMISSIONS ARE RELATED
plt.scatter(df.ENGINESIZE,df.CO2EMISSIONS,color='blue')
plt.title('ENGINESIZE vs CO2EMISSIONS')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

#Now split the dataset into train=80% and test=20% sets using np.random.rand
msk=np.random.rand(len(df))<0.8
train=df[msk]
test=df[~msk]

#Now import LinearRegression from sklearn
from sklearn import linear_model
regr=linear_model.LinearRegression()
#Now create an array using the train data
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
#printing the coefficients
print("The coefficient and intercept are:",regr.coef_,regr.intercept_)

#let's plot the fit line over the train data
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0],color='red')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

#EVALUATION OF THE MODEL with the test set
from sklearn.metrics import r2_score
test_x=np.asanyarray(test[['ENGINESIZE']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
y_pred=regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred- test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , y_pred) )