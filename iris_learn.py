import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from iPython.display import display

from sklearn.datasets import load_iris

# load_iris is a Bunch object, similar to a dictionary
iris_dataset = load_iris()

# Prints the list of keys in the Bunch
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# DESC is a short description of the dataset. 
print(iris_dataset['DESCR'][:193] + "\n...")

# target_names is an array of strings, containing the species of flower we want to predict
print("Target names: {}" .format(iris_dataset['target_names']))

# feature_names is a list of strings, giving description
print("Feature names: \n{}" .format(iris_dataset['feature_names']))

# all of the measurements are stored in a numpy array
print("Type of data: {}" .format(type(iris_dataset['data'])))

# printing the number of rows vs colums can be done with .shape. this array contains measurements for 150 different flowers, or 150 rows, each with 4 values
print("Shape of data: {}" .format(iris_dataset['data'].shape))

print("First five rows of data:\n{}" .format(iris_dataset['data'][:5]))

# 'target is also an array, but really 150 element long vertical vector. it represents the predicted species for each flower.'
print("Type of target: {}" .format(type(iris_dataset['target'])))
print("Shape of target: {}" .format(iris_dataset['target'].shape))

