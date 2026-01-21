from flask import Flask, render_template, request
from model.predictor import WinePredictor

app = Flask(__name__)

# Initialize Predictor
predictor = WinePredictor()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    start_values = {}

    if request.method == 'POST':
        try:
            # Get input values from form
            alcohol = float(request.form['alcohol'])
            flavanoids = float(request.form['flavanoids'])
            color_intensity = float(request.form['color_intensity'])
            hue = float(request.form['hue'])
            od280 = float(request.form['od280'])
            proline = float(request.form['proline'])

            # Store values to repopulate form
            start_values = {
                'alcohol': alcohol,
                'flavanoids': flavanoids,
                'color_intensity': color_intensity,
                'hue': hue,
                'od280': od280,
                'proline': proline
            }

            # Create features list
            features = [alcohol, flavanoids, color_intensity, hue, od280, proline]
            
            # Predict using the class
            prediction_text = predictor.predict(features)

        except ValueError:
            prediction_text = "Error: Invalid input. Please enter numeric values."
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text, start_values=start_values)

if __name__ == '__main__':
    app.run(debug=True)
