import pandas as pd # for data manipulation
import numpy as np # for numerical computations
import matplotlib.pyplot as plt
import seaborn as sns # for data visualization
import scipy as sp # for scientific computing
import sklearn as sk # for machine learning
import missingno as msno # for visualizing missing data






df = pd.read_csv("heart.csv")
#print(df.shape) # rows = 1025, columns = 14

if df.isnull().sum().sum() > 0: # Check for missing values
    df = df.dropna() # Remove missing values

# Remove duplicates
df = df.drop_duplicates()


# one hot encoding for categorical columns
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True) # converts categorical columns into multiple 
#binary (0/1) columns so that machine-learning models can use them correctly.


# Split data into features and target variable 
X = df.drop('target', axis=1)
Y = df['target'] 







