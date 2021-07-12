# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:06:00 2021

@author: Sanjay's
"""

# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y

import numpy as np                       # For all our math needs
n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                        # True labels with noise

import matplotlib.pyplot as plt          # For all our plotting needs
plt.figure()

# Plot the data
plt.scatter(X, y, 12, marker='o')           

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')


# X float(n, ): univariate data
# d int: degree of polynomial  
def polynomial_transform(X, d):
 return np.vander(X,d) #returns the vandermonde matrix 

# Phi float(n, d): transformed data
# y   float(n,  ): labels
def train_model(Phi, y):
  return (np.linalg.inv(np.transpose(Phi).dot(Phi)).dot(np.transpose(Phi).dot(y))) #returns the w matrix with weights

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
  num = len(y)
  y_pred = Phi@w
  err = (y_pred - y)**2
  sum = 0
  for value in err :
      sum += value
  return (sum/num)

w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(list(validationErr.keys()), list(validationErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(testErr.keys()), list(testErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
  X_d = polynomial_transform(x_true, d)
  y_d = X_d @ w[d]
  plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])


# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
  phi = []
  for x in X : 
    z = []
    for b in B:
      z.append(np.exp(-gamma*((x-b)**2)))
    phi.append(z)
  return phi
 
# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter
def train_ridge_model(Phi, y, lam):
  n=len(Phi)
  return (np.linalg.inv((np.transpose(Phi).dot(Phi))+lam*np.eye(n)).dot(np.transpose(Phi).dot(y))) #returns the w vector with weights

w2 = {}               # Dictionary to store all the trained models
validationErr2 = {}   # Validation error of the models
testErr2 = {}         # Test error of all the models
for lam in range(-3, 4, 1):  # Iterate over polynomial degree
    Phi_trn2 = radial_basis_transform(X_trn, X_trn)                 # Transform training data into d dimensions
    w2[lam] = train_ridge_model(Phi_trn2,y_trn,10**lam)                       # Learn model on training data
    
    Phi_val2 = radial_basis_transform(X_val, X_trn)                 # Transform validation data into d dimensions
    validationErr2[lam] = evaluate_model(Phi_val2, y_val, w2[lam])  # Evaluate model on validation data 
    
    Phi_tst2 = radial_basis_transform(X_tst, X_trn)           # Transform test data into d dimensions
    testErr2[lam] = evaluate_model(Phi_tst2, y_tst, w2[lam])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(list(validationErr2.keys()), list(validationErr2.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(testErr2.keys()), list(testErr2.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Lambda(10^x)', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr2.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([-4, 4, 0, 100])

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
lam_range = [10**-3,10**-2,10**-1,1,10**1,10**2,10**3]
for lam in range(-3, 4, 1) :
  X_lam = radial_basis_transform(x_true,X_trn)
  y_lam = X_lam @ w2[lam]
  plt.plot(x_true, y_lam, marker='None', linewidth=2)

plt.legend(['true'] + lam_range)
plt.axis([-8, 8, -15, 15])