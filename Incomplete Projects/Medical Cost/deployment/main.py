from flask import Flask, render_template, request
import joblib
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)

# Load the pre-trained linear regression model
model = joblib.load('deployment\models\insurance_linear_regression.plk')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        sex = object(request.form['sex'])
        smoker = object(request.form['smoker'])
        region = object(request.form['region'])
        
        
        # Predict insurance charges
        prediction = model.predict([['age', 'sex', 'bmi', 'children', 'smoker', 'region']])
        
        # Render the result template with the prediction
        return render_template('home.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)