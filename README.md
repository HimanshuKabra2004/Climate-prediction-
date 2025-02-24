# Climate Prediction Using Linear Regression

## Overview
This project uses machine learning (Linear Regression) to predict future temperatures based on past climate data, including **month, year, and rainfall**. The dataset is continuously updated with user input, improving model accuracy over time.

## Features
- **Historical Data Training:** Uses past climate data for model training.
- **User Data Input:** Allows users to add real-time climate data (month, year, rainfall, and temperature).
- **Dataset Update:** Automatically appends new user data to the existing CSV file.
- **Model Training & Evaluation:** Uses **Linear Regression** and evaluates performance using **Mean Squared Error (MSE)** and **R² Score**.
- **Climate Prediction:** Predicts future temperatures based on user input.
- **Data Visualization:** Plots actual vs predicted temperatures for better understanding.

## Installation
### Prerequisites
Ensure you have **Python 3.x** installed with the following dependencies:
```sh
pip install pandas numpy matplotlib scikit-learn
```

## Usage
### 1. Run the script
```sh
python climate_prediction.py
```

### 2. Input Data
The script prompts users to enter real-time climate data:
```
Enter month (1-12): 5
Enter year: 2024
Enter rainfall amount: 120
Enter actual temperature: 32.5
```

This new data is added to `Temp_and_rain.csv` for future training.

### 3. Model Training & Performance Evaluation
The script trains the **Linear Regression model** using the updated dataset and outputs:
```
Mean Squared Error: 2.45
R^2 Score: 0.89
```

### 4. Predict Future Temperature
Users can provide input for **future predictions**:
```
Enter month for future prediction (1-12): 6
Enter year for future prediction: 2025
Enter expected rainfall amount: 140
Predicted Temperature for given input: 33.2°C
```

## File Structure
```
├── climate_prediction.py  # Main script
├── Temp_and_rain.csv      # Climate dataset (updated dynamically)
├── README.md              # Documentation
```

## Future Enhancements
- Incorporate additional climate factors (humidity, wind speed, etc.).
- Improve model accuracy using advanced ML techniques.
- Develop a **web-based or mobile app** for easy user input and predictions.

## License
This project is open-source under the MIT License.
