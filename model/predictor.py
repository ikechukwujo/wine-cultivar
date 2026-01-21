import os
import joblib
import numpy as np

class WinePredictor:
    def __init__(self, model_dir='model'):
        self.model_path = os.path.join(os.path.dirname(__file__), 'wine_cultivar_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
        self.model = None
        self.scaler = None
        self.load_artifacts()

    def load_artifacts(self):
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("Model and Scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading model/scaler: {e}")
            self.model = None
            self.scaler = None

    def predict(self, features):
        """
        Predicts the wine cultivar based on input features.
        
        Args:
            features (list or np.array): A list of 6 numerical features:
                                         [alcohol, flavanoids, color_intensity, hue, od280, proline]
        
        Returns:
            str: The predicted cultivar name ("Cultivar A", "Cultivar B", "Cultivar C") or an error message.
        """
        if self.model is None or self.scaler is None:
            return "Error: Model not loaded."

        try:
            # maintain compatibility with the original app.py logic which expected a list and converted to 2D array
            input_data = np.array([features])
            
            # Scale features
            features_scaled = self.scaler.transform(input_data)
            
            # Predict
            prediction = self.model.predict(features_scaled)
            cultivar_index = prediction[0]
            
            # Cultivar names mapping
            cultivar_names = {0: "Cultivar A", 1: "Cultivar B", 2: "Cultivar C"}
            result = cultivar_names.get(cultivar_index, f"Class {cultivar_index}")
            
            return f"Predicted Origin: {result}"
            
        except Exception as e:
            return f"Error during prediction: {str(e)}"
