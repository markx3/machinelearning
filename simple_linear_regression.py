# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 00:10:58 2017

@author: marcos
"""

# Simple Linear Regression

# Importando as bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando o dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Dividindo o dataset em training dataset e test dataset

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting SLR ao Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

#Visualizing the Training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.show()
