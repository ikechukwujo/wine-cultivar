from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model and Scaler of wine cultivar
model_path = os.path.join(os.path.dirname(__file__), 'model', 'wine_cultivar_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    if request.method == 'POST':
        if model is None or scaler is None:
             return render_template('index.html', prediction_text="Error: Model not loaded.")

        try:
            # Get input values from form
            alcohol = float(request.form['alcohol'])
            flavanoids = float(request.form['flavanoids'])
            color_intensity = float(request.form['color_intensity'])
            hue = float(request.form['hue'])
            od280 = float(request.form['od280'])
            proline = float(request.form['proline'])

            # Create array for prediction
            features = np.array([[alcohol, flavanoids, color_intensity, hue, od280, proline]])
            
            # Scale features
            features_scaled = scaler.transform(features)

            # Predict
            prediction = model.predict(features_scaled)
            cultivar_index = prediction[0]
            
            # Cultivar names (from dataset)
            cultivar_names = {0: "Cultivar A", 1: "Cultivar B", 2: "Cultivar C"}
            result = cultivar_names.get(cultivar_index, f"Class {cultivar_index}")

            prediction_text = f"Predicted Origin: {result}"

        except ValueError:
            prediction_text = "Error: Invalid input. Please enter numeric values."
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
