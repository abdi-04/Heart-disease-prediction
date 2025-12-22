import pandas as pd # for data manipulation
import numpy as np # for numerical computations
import matplotlib.pyplot as plt
import seaborn as sns # for data visualization
import scipy as sp # for scientific computing
import sklearn as sk # for machine learning


disease_df = pd.read_csv("heart.csv")

#print(disease_df.head()) # Display first few rows of the dataset
#print(disease_df.shape) # rows = 1025, columns = 14