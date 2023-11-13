# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:58:16 2023

@author: avend
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

#file_name ='wine-Quality.csv'
#file_name ='pima-indians-diabetes.csv'
#Función para cargar y dividir los datos de Swedish Auto Insurance Dataset
def read_dataAutoInsur(file_name):
    dataset = pd.read_csv('AutoInsurSweden.csv')
    X = dataset[['X']] #X = number of claims
    y = dataset['Y']  #Y = total payment for all the claims in thousands of Swedish Kronor for geographical zones in Sweden
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Función para cargar y dividir los datos de Wine Quality Dataset
def read_dataWineQuality(file_name):
    dataset = pd.read_csv('wine-Quality.csv', sep=",")
    X = dataset.drop("quality", axis=1)
    y = dataset["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Función para cargar y dividir los datos de pima Indians Diabetes
def read_datapima_Diabetes(file_name):
    dataset = pd.read_csv('pima-indians-diabetes.csv', sep=",")
    X = dataset.drop("Class variable (0 or 1)", axis=1)
    y = dataset["Class variable (0 or 1)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Modelos de regresión
def logistic_Regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Logistic Regression Mean Squared Error:", mse)

def k_Nearest_Neighbors(X_train, X_test, y_train, y_test, n_neighbors=3):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("K-Nearest Neighbors Mean Squared Error:", mse)

def support_Vector_Machine(X_train, X_test, y_train, y_test):
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Support Vector Machine Mean Squared Error:", mse)

def naive_Bayes(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Naive Bayes Mean Squared Error:", mse)

def MLP(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100,50), max_iter=500):
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MLP (Neural Network) Mean Squared Error:", mse)

# # Cargar y dividir los datos
# X_train, X_test, y_train, y_test = read_datapima_Diabetes(file_name)

# # Modelos de regresión
# logistic_Regression(X_train, X_test, y_train, y_test)
# k_Nearest_Neighbors(X_train, X_test, y_train, y_test)
# support_Vector_Machine(X_train, X_test, y_train, y_test)
# naive_Bayes(X_train, X_test, y_train, y_test)
# MLP(X_train, X_test, y_train, y_test)


# Lista de archivos
file_names = ['wine-Quality.csv', 'pima-indians-diabetes.csv', 'AutoInsurSweden.csv']

for file_name in file_names:
    print(f"\nDataset: {file_name}")
    X_train, X_test, y_train, y_test = read_dataWineQuality(file_name)
    X_train, X_test, y_train, y_test = read_datapima_Diabetes(file_name)
    #X_train, X_test, y_train, y_test = read_dataAutoInsur(file_name)
    
    models = [logistic_Regression, k_Nearest_Neighbors, support_Vector_Machine, naive_Bayes, MLP]
    for model in models:
        model(X_train, X_test, y_train, y_test)