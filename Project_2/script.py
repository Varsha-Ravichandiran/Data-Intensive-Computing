# normalization
import numpy as np
import pandas as pd

# normalisation of train_data
df = pd.read_csv('Train.csv')
X = df.drop(['48'], axis=1)
X = (X - np.min(X)) / (np.max(X) - np.min(X)).values
Y = df['48'].values
X[48] = Y
X.to_csv(r'Train_norm.csv', index=False, header=False)

# normalisation of test_data
df1 = pd.read_csv('Test.csv')
X1 = (df1 - np.min(df1)) / (np.max(df1) - np.min(df1)).values
X1.to_csv(r'Test_norm.csv', index=False, header=False)
