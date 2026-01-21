import sys
import os

# Add current directory to path so we can import from model
sys.path.append(os.getcwd())

from model.predictor import WinePredictor

def test_prediction():
    print("Initializing WinePredictor...")
    predictor = WinePredictor()
    
    # Mock data (average values or similar)
    # Alcohol, Flavanoids, Color Intensity, Hue, OD280, Proline
    test_features = [13.5, 2.5, 4.8, 1.0, 3.2, 750]
    
    print(f"Testing with features: {test_features}")
    result = predictor.predict(test_features)
    print(f"Result: {result}")
    
    if "Predicted Origin" in result or "Class" in result:
        print("SUCCESS: Prediction returned valid format.")
    else:
        print("FAILURE: Prediction returned unexpected format or error.")

if __name__ == "__main__":
    test_prediction()
