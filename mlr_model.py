import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("data.csv")

# Step 2: Check for Missing Values
print("Missing Values:\n", df.isnull().sum())

# Step 3: Define Independent (X) and Dependent (y) Variables
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Step 4: Split Data into Training and Testing Sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared Score:", r2)

# Step 8: Predict for New Data
new_house_df = pd.DataFrame([[2300, 3, 15]], columns=['Area', 'Bedrooms', 'Age'])
predicted_price = model.predict(new_house_df)


print("\nPredicted Price for new house:", predicted_price[0])
