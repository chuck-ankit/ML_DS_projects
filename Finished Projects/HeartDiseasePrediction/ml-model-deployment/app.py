from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# Load the pre-trained model
model_path = 'BeatWise.plk'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_heart_diseases():
    # Get user inputs from the form
    Name = request.form.get('Name')
    Age = int(request.form.get('Age'))
    Sex = int(request.form.get('Sex'))
    cp = int(request.form.get('cp'))
    trestbps = int(request.form.get('trestbps'))
    chol = int(request.form.get('chol'))
    fbs = int(request.form.get('fbs'))
    restecg = int(request.form.get('restecg'))
    thalach = int(request.form.get('thalach'))
    exang = int(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slope = int(request.form.get('slope'))
    ca = int(request.form.get('ca'))
    thal = int(request.form.get('thal'))

    # Make a prediction using the loaded model
    input_data = np.array([Age, Sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, 13)
    result = model.predict(input_data)

    # Map numeric result to labels
    if result[0] == 0:
        prediction_label = "Low"
    elif result[0] == 1:
        prediction_label = "High"
    else:
        prediction_label = "Unknown"

    # Render a new template with the prediction result
    return render_template('result.html', Name=Name, prediction=prediction_label)

if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)
    # app.run(host='0.0.0.0',port=8080)