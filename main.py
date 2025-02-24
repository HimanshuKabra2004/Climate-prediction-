import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load existing dataset
data = pd.read_csv('Temp_and_rain.csv')

# Get new input data from user
month = int(input("Enter month (1-12): "))
year = int(input("Enter year: "))
rain = float(input("Enter rainfall amount: "))
tem = float(input("Enter actual temperature: "))

# Append new data to the dataset
new_entry = pd.DataFrame({'Month': [month], 'Year': [year], 'rain': [rain], 'tem': [tem]})
data = pd.concat([data, new_entry], ignore_index=True)

# Save the updated dataset back to the CSV file
data.to_csv('Temp_and_rain.csv', index=False)

# Prepare data for training
X = data[['Month', 'Year', 'rain']]
y = data['tem']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.show()

# Predict future climate
target_month = int(input("Enter month for future prediction (1-12): "))
target_year = int(input("Enter year for future prediction: "))
target_rain = float(input("Enter expected rainfall amount: "))

future_data = np.array([[target_month, target_year, target_rain]])
future_prediction = model.predict(future_data)
print(f"Predicted Temperature for given input: {future_prediction[0]:.2f}°C")
