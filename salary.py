import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
dataset=pd.read_csv("salary_data.csv")
print(dataset.head())
print(dataset.shape)
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
print ("x",x)
print ("y",y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x)
print(y)
print("x test",x_test)
print("x_train",x_train)
print("y test",y_test)
print("y train",y_train)
#scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
print("x_test standard",x_test)
print("x_train standard",x_train)
#fit-learning
#stadardisation means standard deviation =1 and mean=0.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print("y_pred",y_pred)
#graph
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("salary vs experiance(training)")
plt.xlabel("year of experiance")
plt.ylabel("salary")
plt.show()
#training graph
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="blue")
plt.title("salary vs experiance(testing)")
plt.xlabel("year of experiance")
plt.ylabel("salary")
plt.show()

print(regressor.score(x_test,y_test))
y_result=regressor.predict([[15]])
print(y_result)