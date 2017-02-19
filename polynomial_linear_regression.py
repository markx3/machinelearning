# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 15:42:59 2017

@author: marcos
"""

# Polynomial Linear Regression

# Pre-processamento de dados

# Importando as bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando o dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Dividindo o dataset em training dataset e test dataset
''' sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) '''
# Poucas observações -> não dividir em training/test

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg = lin_reg.fit(X, Y)

# Fitting Polynomial Linear Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg = poly_reg.fit(X, Y)
