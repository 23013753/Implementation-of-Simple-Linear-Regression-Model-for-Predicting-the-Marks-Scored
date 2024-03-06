# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
 1.Import the standard Libraries.
 2.Set variables for assigning dataset values.
 3.Import linear regression from sklearn.
 4.Assign the points for representing in the graph.
 5.Predict the regression for marks by using the representation of the graph.
 6.Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: vishal .s 
RegisterNumber: 212223240184
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
```

## Output:
# Dataset:
![image](https://github.com/23013753/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145634121/42922c96-19fa-4e52-8210-5181559d16b0)


# Head Values:
![image](https://github.com/23013753/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145634121/97e1071d-8767-4e88-a306-47fd61023312)

# Tail Values:

![image](https://github.com/23013753/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145634121/b10726f8-cb88-47ba-897f-cf68270d1fab)

# X and Y Values:

![image](https://github.com/23013753/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145634121/6edd4a89-a931-44fa-bdce-07965446f931)

# Predication values of X and Y:

![image](https://github.com/23013753/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145634121/66fca703-916e-45d0-96eb-41e6ae385912)

# MSE,MAE and RMSE:

![image](https://github.com/23013753/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145634121/a692e7ee-4835-4468-8e2b-d9c039151260)

# Training Set:

![image](https://github.com/23013753/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145634121/6e63651b-08c5-4f7b-a0d5-fb70d64f4d66)

# Testing Set:

![image](https://github.com/23013753/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145634121/e7545158-d8de-429c-9312-4750f460e2e0)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
