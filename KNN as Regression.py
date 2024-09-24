import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load the dataset (with headers)
dataCSV = pd.read_csv('car_evaluation.csv', header=0)

# Add column names
dataCSV.columns = ['buying_price', 'maintenance_cost', 'number_of_doors', 'number_of_persons', 'lug_boot', 'safety', 'decision']

# Initialize LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to each categorical column
dataCSV['buying_price'] = le.fit_transform(dataCSV['buying_price'])
dataCSV['maintenance_cost'] = le.fit_transform(dataCSV['maintenance_cost'])
dataCSV['number_of_doors'] = le.fit_transform(dataCSV['number_of_doors'])
dataCSV['number_of_persons'] = le.fit_transform(dataCSV['number_of_persons'])
dataCSV['lug_boot'] = le.fit_transform(dataCSV['lug_boot'])
dataCSV['safety'] = le.fit_transform(dataCSV['safety'])
dataCSV['decision'] = le.fit_transform(dataCSV['decision'])  # Target variable

# Features (X) - independent variables
X = dataCSV[['buying_price', 'maintenance_cost', 'number_of_doors', 'number_of_persons', 'lug_boot', 'safety']].values

# Target (Y) - dependent variable (encoded 'decision' column)
Y = dataCSV['decision'].values

# Check the shape of the dataset
print(f"Dataset shape: {dataCSV.shape}")

# Initialize KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors=10)

# Fit the model with the data
knnr.fit(X, Y)

# Make predictions
Y_pred = knnr.predict(X)

# Calculate Mean Squared Error
mse = mean_squared_error(Y, Y_pred)

# Print the MSE
print(f"The MSE is: {mse}")
