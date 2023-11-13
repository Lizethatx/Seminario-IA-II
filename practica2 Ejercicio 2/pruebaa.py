# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:54:40 2023

@author: avend
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

#leer dataset de Logistic Regression
def read_dataLR(AutoInsurSweden):
    dataset = pd.read_csv('AutoInsurSweden.csv')
    X = dataset[['X']] #X = number of claims
    y = dataset['Y']  #Y = total payment for all the claims in thousands of Swedish Kronor for geographical zones in Sweden
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def logistic_Regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def k_Nearest_Neighbors(X_train, y_train, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def support_Vector_Machine(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model

def naive_Bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def MLP(X_train, y_train, hidden_layer_sizes=(100,50), max_iter=500):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


# Cargar datos usando la función read_dataLR
X_train, X_test, y_train, y_test = read_dataLR('AutoInsurSweden.csv')

# Logistic Regression
lr_model = logistic_Regression(X_train, y_train)
lr_accuracy = lr_model.score(X_test, y_test)
print("Logistic Regression Accuracy:", lr_accuracy)

# K-Nearest Neighbors
knn_model = k_Nearest_Neighbors(X_train, y_train)
knn_accuracy = knn_model.score(X_test, y_test)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)

# Support Vector Machine
svm_model = support_Vector_Machine(X_train, y_train)
svm_accuracy = svm_model.score(X_test, y_test)
print("Support Vector Machine Accuracy:", svm_accuracy)

# Naive Bayes
nb_model = naive_Bayes(X_train, y_train)
nb_accuracy = nb_model.score(X_test, y_test)
print("Naive Bayes Accuracy:", nb_accuracy)

# MLP (Neural Network)
mlp_model = MLP(X_train, y_train)
mlp_accuracy = mlp_model.score(X_test, y_test)
print("MLP (Neural Network) Accuracy:", mlp_accuracy)
