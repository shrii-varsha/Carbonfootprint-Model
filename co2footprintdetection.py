import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import warnings 

sb.set(context="notebook", palette="Spectral", style="darkgrid", font_scale=1, color_codes=True)
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv(r"C:\Users\Shrii\Downloads\synthetic_gas_data_150.csv")  
# Select relevant columns
data_cols = data[['AQI', 'CO2', 'CH4', 'CO', 'H', 'N2O', 'O3', 'CFC']]
print(data.describe())
print(data_cols.head())

# Check for linearity in the data
pairplot = sb.pairplot(data_cols)
plt.show()  # Uncomment to display the pairplot

# Define features and target variable
X = data[['CO2', 'CH4', 'CO', 'H', 'N2O', 'O3', 'CFC']]
y = data['AQI']

# Standardize the features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Fit the linear regression model
LR_model = linear_model.LinearRegression()
LR_model.fit(X_scaled, y)
# thresholds = {
#     'CO2': 400,    
#     'CH4': 1900,   
#     'CO': 9,       
#     'H': 1,        
#     'N2O': 300,    
#     'O3': 0.070,    
#     'CFC': 0.1      
# }

# Input from user
co2 = float(input("Enter CO2 level: "))
ch4 = float(input("Enter CH4 level: "))
co = float(input("Enter CO level: "))
h = float(input("Enter H level: "))
n2o = float(input("Enter N2O level: "))
o3 = float(input("Enter O3 level: "))
cfc = float(input("Enter CFC level: "))

# Prepare input for prediction
input_data = np.array([[co2, ch4, co, h, n2o, o3, cfc]])
input_scaled = sc.transform(input_data)

# Predict AQI
aqi_pred = LR_model.predict(input_scaled)
print("Predicted AQI Value:", aqi_pred[0])

# Residuals or error calculation
residuals = y.values - LR_model.predict(X_scaled)
mean_residuals = np.mean(residuals)
print("Mean residual = ", mean_residuals)

# Finding R2 score to assess model accuracy
print("R2 score = ", r2_score(y_true=y, y_pred=LR_model.predict(X_scaled)))

# Normality check of residuals
normality_curve = sb.displot(residuals, kde=True)
plt.show()  # Uncomment to display the density plot

# Multi-collinearity check
collinearity_check = data_cols.corr()
print(collinearity_check)

# Heatmap to show correlation status
plt.figure(figsize=(10, 8))
heatMapFig = sb.heatmap(collinearity_check, annot=True, cmap="Reds", square=True)
plt.show()  # Uncomment to display the heatmap

print("Done")