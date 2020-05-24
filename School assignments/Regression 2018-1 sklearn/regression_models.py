# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#Importing dataset and cleaning 'nan' columns and rows
dataset = pd.read_csv('ENB2012_data.csv')
dataset = dataset.drop('Unnamed: 10', axis = 1)
dataset = dataset.drop('Unnamed: 11', axis = 1)
dataset = dataset[:767]

#creating test and train sets
X = dataset.iloc[:, :-2].values
y1 = dataset.iloc[:, 8].values
y2 = dataset.iloc[:, 9].values

#Doing Grid search for every model
#Grid Search function
def grid_search_func (param_grid, reg_model, y):
    grid_search = GridSearchCV(reg_model, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    print(grid_search.best_params_)


#Linear Regression
param_grid_linear = [{'fit_intercept': [True, False], 'normalize': [True, False]}]
regressor_linear = LinearRegression()
grid_search_func(param_grid_linear, regressor_linear, y1)
grid_search_func(param_grid_linear, regressor_linear, y2)


#Decision Tree Regression
param_grid_tree = [{'criterion': ['mse'], 'splitter': ['best', 'random'], 
                    'max_depth': [3, 5, 8, 10], 'max_features': [2, 4, 6, 8, 'sqrt']}]
regressor__tree = DecisionTreeRegressor()
grid_search_func(param_grid_tree, regressor__tree, y1)
grid_search_func(param_grid_tree, regressor__tree, y2)


#Random Forest Regression
param_grid_forest = [{'n_estimators': [3, 10, 30, 100], 'max_features': [2, 4, 6, 8, 'sqrt']}, 
                    {'bootstrap': [False], 'n_estimators': [3, 10, 100], 'max_features': [2, 3,4, 'sqrt']}]
regressor_forest = RandomForestRegressor()
grid_search_func(param_grid_forest, regressor_forest, y1)
grid_search_func(param_grid_forest, regressor_forest, y2)


#K-Nearest Neighbors Regression
param_grid_knn = [{'n_neighbors': [3, 5, 10, 20, 30, 40, 50], 'weights': ['uniform', 'distance']}]
regressor_knn = KNeighborsRegressor()
grid_search_func(param_grid_knn, regressor_knn, y1)
grid_search_func(param_grid_knn, regressor_knn, y2)

#Support Vector Regression
param_grid_svr = [{'kernel': ['rbf'], 'C': [1.5, 10], 'gamma': [1e-7, 1e-4], 'epsilon': [0.1, 0.2, 0.3, 0.5]}]
regressor_svr = SVR()
grid_search_func(param_grid_svr, regressor_svr, y1)
grid_search_func(param_grid_svr, regressor_svr, y2)


#Defining parameters for every model

#Linear Regression
regressor_linear_y1 = LinearRegression(fit_intercept = True, normalize = False)
regressor_linear_y2 = LinearRegression(fit_intercept = True, normalize = True)
#Decision Tree Regression (parameters are same for y1 and y2)
regressor__tree_y = DecisionTreeRegressor(criterion = 'mse', max_depth = 10, max_features = 8, splitter = 'best')
#Random Forest Regression
regressor_forest_y1 = RandomForestRegressor(max_features = 8, n_estimators = 30)
regressor_forest_y2 = RandomForestRegressor(max_features = 2, n_estimators = 30)
#K-Nearest Neighbors Regression
regressor_knn_y1 = KNeighborsRegressor(n_neighbors = 5, weights = 'uniform')
regressor_knn_y2 = KNeighborsRegressor(n_neighbors = 3, weights = 'uniform')
#Support Vector Regression
regressor_svr_y1 = SVR(C = 10, epsilon = 0.5, gamma =  0.0001, kernel = 'rbf')
regressor_svr_y2 = SVR(C = 10, epsilon = 0.2, gamma =  0.0001, kernel = 'rbf')

#Cross-validation functions
def display_scores(scores): 
    print("Scores:", scores)
    print("Mean:", scores.mean()) 
    print("Standard deviation:", scores.std())


def cross_validation(reg_model, y):
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    lin_scores = cross_val_score(reg_model, X, y, scoring="neg_mean_squared_error", cv=kf) 
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)
    
#Cross-validation for y1 and y2
#Linear Regression
cross_validation(regressor_linear_y1, y1)
cross_validation(regressor_linear_y2, y2)
#Decision Tree Regression
cross_validation(regressor__tree_y, y1)
cross_validation(regressor__tree_y, y2)
#Random Forest Regression
cross_validation(regressor_forest_y1, y1)
cross_validation(regressor_forest_y2, y2)
#K-Nearest Neighbors Regression
cross_validation(regressor_knn_y1, y1)
cross_validation(regressor_knn_y2, y2)
#Support Vector Regression
cross_validation(regressor_svr_y1, y1)
cross_validation(regressor_svr_y2, y2)