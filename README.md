![Screenshot 2024-05-19 115859](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147473858/fdd4aab8-42eb-49a7-8fc8-78a51d45cebb)# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:


1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Visulaize the data and define the sigmoid function, cost function and gradient descent.

4.Import linear regression from sklearn.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Obtain the graph.


## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 23013598
RegisterNumber:S.PREM KUMAR

``` 
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset['gender'] = dataset['gender'].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta ,X ,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iters):
    m=len(y)
    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -=alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iters=1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```

## Output:
!![Screenshot 2024-05-19 115812](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147473858/27aae0ad-85c6-48ee-804f-e3949579ce2b)
!(![Screenshot 2024-05-19 115821](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147473858/da32d8b6-70f7-428a-841a-8b655e87f591)
)
!(![Screenshot 2024-05-19 115843](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147473858/973ba095-e3cd-473f-9d27-7d98791dd708)
)
!(![Screenshot 2024-05-19 115859](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147473858/e321aa43-5e52-4d8d-9d18-92358f0e974e)
)
!(![Screenshot 2024-05-19 115905](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147473858/a2af5b55-adb2-4cb0-8064-28e5ff252376)
)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

