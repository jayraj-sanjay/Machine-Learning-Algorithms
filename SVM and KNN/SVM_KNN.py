# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:46:09 2021

@author: Sanjay's
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
  # Generate a non-linear data set
  X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)
   

  m = 30
  np.random.seed(30)  # Deliberately use a different seed
  ind = np.random.permutation(n_samples)[:m]
  X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
  y[ind] = 1 - y[ind]

  # Plot this data
  cmap = ListedColormap(['#b30065', '#178000'])  
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')       
  
  # First, we use train_test_split to partition (X, y) into training and test sets
  X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, 
                                                random_state=42)

  # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
  X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, 
                                                random_state=42)
  
  return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)


def visualize(models, param, X, y):
  # Initialize plotting
  if len(models) % 3 == 0:
    nrows = len(models) // 3
  else:
    nrows = len(models) // 3 + 1
    
  fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
  cmap = ListedColormap(['#b30065', '#178000'])

  # Create a mesh
  xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
  yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
  xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01), 
                             np.arange(yMin, yMax, 0.01))

  for i, (p, clf) in enumerate(models.items()):
    # if i > 0:
    #   break
    r, c = np.divmod(i, 3)
    ax = axes[r, c]

    # Plot contours
    zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
    zMesh = zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

    if (param == 'C' and p > 0.0) or (param == 'gamma'):
      ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1], 
                 alpha=0.5, linestyles=['--', '-', '--'])

    # Plot data
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')       
    ax.set_title('{0} = {1}'.format(param, p))
    
    # Generate the data
n_samples = 300    # Total size of data set 
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)

# Learn support vector classifiers with a radial-basis function kernel with 
# fixed gamma = 1 / (n_features * X.std()) and different values of C
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error,accuracy_score

C_range = np.arange(-3.0, 6.0, 1.0)
C_values = np.power(10.0, C_range) 

models = dict()
trnErr = dict()
valErr = dict()

for C in C_values:
  #using the SVC below with varying C values and gamma is set to scale 
  clf = SVC(C,gamma='scale')
  models[C]= clf.fit(X_trn, y_trn)
  ypred_val = models[C].predict(X_val)
  ypred_trn = models[C].predict(X_trn)

  #calculating mean squared error for validation and training data
  valErr[C]=mean_squared_error(y_val, ypred_val )
  trnErr[C]=mean_squared_error(y_trn, ypred_trn )

#plotting the graph for varying C values with error for training and validation
plt.figure()
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Regularization Parameter C', fontsize=16)
plt.ylabel('Validation/Training error', fontsize=16)
plt.xticks(list(valErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Training Error'], fontsize=16)
plt.xscale('log')

  
visualize(models, 'C', X_trn, y_trn)

#Calculating the difference between validation error and training error
err_diff = dict()
for key in valErr:
  err_diff[key]=abs(trnErr[key]-valErr[key])

#Find the minimum value in the difference of the error to get the best C value
c_best=min(err_diff, key=lambda k: err_diff[k])

#predicting the y values for training data with the model for best C value
ypred_tst = models[c_best].predict(X_tst)

#Calculating the accuracy by comparing the given y values and the predicted y values of the training data
accuracy = accuracy_score(y_tst,ypred_tst)
print("Best value for C:", c_best)
print("Accuracy percentage for the test set with the best C value: ",round(accuracy*100,2))

# Learn support vector classifiers with a radial-basis function kernel with 
# fixed C = 10.0 and different values of gamma
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()

for G in gamma_values:
  #using the SVC below with varying gamma values and C value is set to 10.0
  clf = SVC(C=10.0,gamma=G)
  models[G]= clf.fit(X_trn, y_trn)
  ypred_val = models[G].predict(X_val)
  ypred_trn = models[G].predict(X_trn)

  #calculating mean squared error for validation and training data
  
  valErr[G]=mean_squared_error(y_val, ypred_val )
  trnErr[G]=mean_squared_error(y_trn, ypred_trn )

#plotting the graph for varying gamma values with error for training and validation 
plt.figure()
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('RBF kernel parameter gamma', fontsize=16)
plt.ylabel('Validation/Training error', fontsize=16)
plt.xticks(list(valErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Training Error'], fontsize=16)
plt.xscale('log')
  
visualize(models, 'gamma', X_trn, y_trn)

#Calculating the difference between validation error and training error
err_diff = dict()
for key in valErr:
  err_diff[key]=abs(trnErr[key]-valErr[key])

#Find the minimum value in the difference of the error to get the best gamma value
gamma_best=min(err_diff, key=lambda k: err_diff[k])

#predicting the y values for training data with the model for best gamma value
ypred_tst = models[gamma_best].predict(X_tst)

#Calculating the accuracy by comparing the given y values and the predicted y values of the training data
accuracy = accuracy_score(y_tst,ypred_tst)
print("Best value for gamma:", gamma_best)
print("Accuracy percentage for the test set with the best gamma value: ",round(accuracy*100,2))

# Load the Breast Cancer Diagnosis data set; download the files from eLearning
# CSV files can be read easily using np.loadtxt()
import pandas as pd 

# TRAINING DATA
train_data = pd.read_csv('wdbc_trn.csv',header=None)
y_trn_wbc = train_data.iloc[:,0].to_numpy()
X_trn_wbc = train_data.iloc[:,1:].to_numpy()

# VALIDATION DATA
val_data = pd.read_csv('wdbc_val.csv',header=None)
y_val_wbc = val_data.iloc[:,0].to_numpy()
X_val_wbc = val_data.iloc[:,1:].to_numpy()

# TEST DATA
tst_data = pd.read_csv('wdbc_tst.csv',header=None)
y_tst_wbc = tst_data.iloc[:,0].to_numpy()
X_tst_wbc = tst_data.iloc[:,1:].to_numpy()


#set of different C values
C_range = np.arange(-2.0,5.0, 1.0)
C_values = np.power(10.0, C_range)

#set of different gamma values
gamma_range = np.arange(-3.0, 3.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()
val_accuracy = dict()
tst_accuracy = dict()
best_c_gamma = []

for C in C_values:
  #iterating for each C value all the gamma values below
  for G in gamma_values:
    clf = SVC(C=C,gamma=G)
    models[C,G]= SVC(C=C,gamma=G).fit(X_trn_wbc, y_trn_wbc)
    ypred_val = models[C,G].predict(X_val_wbc)
    ypred_trn = models[C,G].predict(X_trn_wbc)
    ypred_tst = models[C,G].predict(X_tst_wbc)

    #calculating mean squared error for validation and training data
    valErr[C,G]=mean_squared_error(y_val_wbc, ypred_val )
    trnErr[C,G]=mean_squared_error(y_trn_wbc, ypred_trn )
    #Calculating validation accuracy
    val_accuracy[C,G] = accuracy_score(y_val_wbc,ypred_val)

#print training error
print('training error')
for cg in trnErr:
  print(cg,':',trnErr[cg])
#print Valodation error
print('Validation error')
for cg in valErr:
  print(cg,':',valErr[cg])

#finding all the keys with maximum accuracy in the validation set
maxAccuracy= max(val_accuracy.values())
for cg in val_accuracy:
  if(val_accuracy[cg]>=maxAccuracy):
    best_c_gamma.append(cg)

#Calculating each test accuracy for best c and gamma values
for cg in best_c_gamma:
  ypred_tst = models[cg].predict(X_tst_wbc)
  tst_accuracy[cg]=accuracy_score(y_tst_wbc,ypred_tst)

#Finding the max test accuracy and its respective key
maxTstAccuracy = max(tst_accuracy.values())
cg_best=max(tst_accuracy, key=lambda k: tst_accuracy[k])

print("Best values for C and Gamma are:", cg_best)
print("Accuracy percentage for the test set with the best c and gamma value: ", round(maxTstAccuracy*100,2))

from sklearn.neighbors import KNeighborsClassifier

k_values = [1,5,11,15,21]

models = dict()
trnErr = dict()
valErr = dict()
val_accuracy = dict()
tst_accuracy = dict()
best_k = []

for K in k_values:
  #using KNeighborsClassifier with different values of K
  neigh = KNeighborsClassifier(n_neighbors=K,algorithm='kd_tree')
  models[K]=neigh.fit(X_trn_wbc, y_trn_wbc)
  ypred_val = models[K].predict(X_val_wbc)
  ypred_trn = models[K].predict(X_trn_wbc)
  
  #Calculating the validation and training error 
  valErr[K]=mean_squared_error(y_val_wbc, ypred_val)
  trnErr[K]=mean_squared_error(y_trn_wbc, ypred_trn)
  #Calculating validation accuracy
  val_accuracy[K] = accuracy_score(y_val_wbc,ypred_val)

#plotting the graph for varying gamma values with error for training and validation 
plt.figure()
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Different values of K', fontsize=16)
plt.ylabel('Validation/Training error', fontsize=16)
plt.xticks(list(valErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Training Error'], fontsize=16)

#finding all the keys with maximum accuracy in the validation set
maxAccuracy= max(val_accuracy.values())
for k in val_accuracy:
  if(val_accuracy[k]>=maxAccuracy):
    best_k.append(k)

#Calculating each test accuracy for best k values
for k in best_k:
  ypred_tst = models[k].predict(X_tst_wbc)
  tst_accuracy[k]=accuracy_score(y_tst_wbc,ypred_tst)

#Finding the max test accuracy and its respective key
maxTstAccuracy = max(tst_accuracy.values())
k_best=max(tst_accuracy, key=lambda k: tst_accuracy[k])

#Calculating the accuracy by comparing the given y values and the predicted y values of the training data
accuracy = accuracy_score(y_tst_wbc,ypred_tst)
print("Best values for k are:", k_best)
print("Accuracy percentage for the test set with the best k value: ", round(maxTstAccuracy*100,2))



























































