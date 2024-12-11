# -*- coding: utf-8 -*-
"""Final Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1acAVU4Vz3zUNfTAWtYlaAoJyzcJmzUtg
"""

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential # type: ignore
from keras.layers import Dense, LSTM # type: ignore
import pickle

# Step 2: Load the Dataset
file_path = 'data/CoffeeShopSales.xlsx'  # Update with the correct path
data = pd.read_excel(file_path)

# Step 3: Data Cleaning
# Check for missing values
print(data.isnull().sum())

# Fill missing values or drop rows/columns if needed
data = data.dropna()

# Handle outliers (example for 'unit_price')
q1 = data['unit_price'].quantile(0.25)
q3 = data['unit_price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data = data[(data['unit_price'] >= lower_bound) & (data['unit_price'] <= upper_bound)]

# Convert 'transaction_date' to datetime and extract useful features
data['transaction_date'] = pd.to_datetime(data['transaction_date'])
data['day_of_week'] = data['transaction_date'].dt.day_name()
data['month'] = data['transaction_date'].dt.month
data['hour'] = pd.to_datetime(data['transaction_time'], format='%H:%M:%S').dt.hour

# Step 4: Data Exploration
# Plot sales trends
sales_trend = data.groupby('transaction_date')['transaction_qty'].sum()
plt.figure(figsize=(12, 6))
plt.plot(sales_trend)
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# Day of the week analysis
day_sales = data.groupby('day_of_week')['transaction_qty'].sum().sort_values()
day_sales.plot(kind='bar', figsize=(10, 5), title='Sales by Day of the Week')
plt.ylabel('Total Sales')
plt.show()

# Top products
top_products = data.groupby('product_type')['transaction_qty'].sum().sort_values(ascending=False)
top_products.plot(kind='bar', figsize=(10, 5), title='Top Products by Quantity Sold')
plt.ylabel('Total Quantity Sold')
plt.show()

# Step 5: Feature Engineering
data['revenue'] = data['transaction_qty'] * data['unit_price']
features = ['transaction_qty', 'unit_price', 'hour', 'day_of_week', 'month']
data = pd.get_dummies(data, columns=['day_of_week'], drop_first=True)

# Step 6: Model Selection and Training
# Prepare data for regression
X = data.drop(['transaction_id', 'transaction_date', 'transaction_time', 'store_id',
               'store_location', 'product_id', 'product_category', 'product_type',
               'product_detail', 'revenue'], axis=1)
y = data['revenue']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Step 7: Evaluate Models
def evaluate_model(model_name, y_true, y_pred):
    print(f"Model: {model_name}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.2f}")
    return r2_score(y_true, y_pred)  # Return the R2 score for saving

# Evaluate models
lr_r2_score = evaluate_model("Linear Regression", y_test, y_pred_lr)
rf_r2_score = evaluate_model("Random Forest", y_test, y_pred_rf)

# Save Model Scores
model_scores = {
    "Linear Regression": lr_r2_score,
    "Random Forest": rf_r2_score,
    # Optionally add LSTM if desired
}

# LSTM for Sequential Data
X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_seq, y_train, epochs=10, batch_size=32, verbose=1)

y_pred_lstm = model.predict(X_test_seq)
evaluate_model("LSTM", y_test, y_pred_lstm)

# Step 8: Save Models with Pickle
with open('models/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('models/model_scores.pkl', 'wb') as f:
    pickle.dump(model_scores, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)