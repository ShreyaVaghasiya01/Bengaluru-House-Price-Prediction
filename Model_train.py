import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load Data
df = pd.read_csv("Bengaluru_House_Data.csv")

# Rename for consistency if needed
if 'size' in df.columns:
    df.rename(columns={'size': 'Size'}, inplace=True)
if 'price' in df.columns:
    df.rename(columns={'price': 'Price_lakhs'}, inplace=True)

# Remove NaNs from key columns
df = df.dropna(subset=['total_sqft', 'Price_lakhs'])

# Add fallback values
df['Size_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
df = df.dropna(subset=['Size_sqft'])

print(df.head())

#define X and y
X=df[["area_type","location","Size","Size_sqft","bath","balcony","availability"]]
y=df["Price_lakhs"]

#encode categorical columns
X = pd.get_dummies(X, columns=["area_type","location","Size","availability"], drop_first=True)

# Fill any remaining NaN values with median (numeric columns)
X = X.fillna(X.median())

#split data into train and test
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#train the model
model=LinearRegression()
model.fit(X_train,y_train)

#make predictions
y_pred=model.predict(x_test)

#evaluate the model
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Save the model
joblib.dump(model, 'house_price_model_lr.pkl')











