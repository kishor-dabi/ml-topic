# Decision Tree Regression
# -*- coding: utf-8 -*-
# Regression template
"""
Created on Fri Apr 10 12:01:43 2020

@author: kishoredabi
"""



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# !----------------- data set is to sort so comment this
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling  # ----------------- for some model don't need or inbuilt 
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)


# fitting  Desicion tree regression  model to tha dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# # predicting a new result with linear regressioin
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))
# y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[7.5]]))))

# Visualising the Desicion tree regression result
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff( Desicion tree regression)')
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()

# Visualising the  Desicion tree regression result (for high resolution and smoth curve)
X_grid = np.arange(min(X), max(X), 0.01 )
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff( Desicion tree regression )')
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()

# # predicting a new result with  Desicion tree regression 
# lin_reg.predict(np.array([6.5]).reshape(1, 1))
# # predicting a new result with Desicion tree 
# lin_reg_2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, 1)))


# -------------------------------------------------------end by me
# # Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # Importing the dataset
# dataset = pd.read_csv('Position_Salaries.csv')
# X = dataset.iloc[:, 1:2].values
# y = dataset.iloc[:, 2].values

# # Splitting the dataset into the Training set and Test set
# """from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# # Feature Scaling
# """from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)"""

# # Fitting Decision Tree Regression to the dataset
# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state = 0)
# regressor.fit(X, y)

# # Predicting a new result
# y_pred = regressor.predict(6.5)

# # Visualising the Decision Tree Regression results (higher resolution)
# X_grid = np.arange(min(X), max(X), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color = 'red')
# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
# plt.title('Truth or Bluff (Decision Tree Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()