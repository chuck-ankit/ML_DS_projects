from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained linear regression model
model = joblib.load('models/insurance_linear_regression.plk')

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
        sex = request.form['sex']
        smoker = request.form['smoker']
        region = request.form['region']
        
        # Convert categorical variables into numerical values
        sex = 1 if sex.lower() == 'male' else 0
        smoker = 1 if smoker.lower() == 'yes' else 0
        
        # Transform region into dummy variables
        if region.lower() == 'southwest':
            region_southwest = 1
            region_southeast = 0
            region_northeast = 0
            region_northwest = 0
        elif region.lower() == 'southeast':
            region_southwest = 0
            region_southeast = 1
            region_northeast = 0
            region_northwest = 0
        elif region.lower() == 'northeast':
            region_southwest = 0
            region_southeast = 0
            region_northeast = 1
            region_northwest = 0
        else:  # assume 'northwest' as default
            region_southwest = 0
            region_southeast = 0
            region_northeast = 0
            region_northwest = 1
        
        # Predict insurance charges
        prediction = model.predict([[age, sex, bmi, children, smoker, region_southwest, region_southeast, region_northeast, region_northwest]])
        
        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
