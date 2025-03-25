from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and train model
df = pd.read_csv("data.csv")
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    age = int(request.form['age'])

    new_house_df = pd.DataFrame([[area, bedrooms, age]], columns=['Area', 'Bedrooms', 'Age'])
    predicted_price = model.predict(new_house_df)[0]

    return render_template('index.html', prediction=f"Predicted House Price: ₹{predicted_price:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)
