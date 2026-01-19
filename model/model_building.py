import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load the Wine dataset
print("Loading Wine dataset...")
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['cultivar'] = wine.target

# 2. Feature Selection
# Selected features:
# 1. alcohol
# 2. flavanoids
# 3. color_intensity
# 4. hue
# 5. od280/od315_of_diluted_wines
# 6. proline
selected_features = [
    'alcohol',
    'flavanoids',
    'color_intensity',
    'hue',
    'od280/od315_of_diluted_wines',
    'proline'
]

print(f"Selected features: {selected_features}")
X = df[selected_features]
y = df['cultivar']

# 3. Data Preprocessing
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the Model (Random Forest Classifier)
print("Training Random Forest Classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# 5. Evaluate the Model
print("Evaluating model...")
y_pred = rf_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=wine.target_names)

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# 6. Save the Model and Scaler
# We need to save the scaler too to preprocess new inputs correctly
model_path = 'model/wine_cultivar_model.pkl'
scaler_path = 'model/scaler.pkl'

print(f"Saving model to {model_path}...")
joblib.dump(rf_classifier, model_path)
print(f"Saving scaler to {scaler_path}...")
joblib.dump(scaler, scaler_path)

print("Model development complete.")
