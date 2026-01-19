# Wine Cultivar Prediction System

## Overview
This project is a Machine Learning-powered web application designed to predict the cultivar origin of wines based on their chemical composition. It utilizes a **Random Forest Classifier** trained on the famous Wine dataset from the UCI Machine Learning Repository.

The system features a user-friendly web interface built with **Flask**, allowing users to input chemical properties and receive instant predictions about the wine's cultivar (Cultivar A, B, or C).

## Features
- **Machine Learning Model**: Accurate classification using Random Forest.
- **Web Interface**: Clean and responsive UI for easy interaction.
- **Real-time Prediction**: Instant results upon form submission.
- **Scalable Design**: Modular code structure separating model logic from the web application.

## Dataset & Features
The model is trained on the following chemical properties selected for their predictive power:
1. **Alcohol**
2. **Flavanoids**
3. **Color Intensity**
4. **Hue**
5. **OD280/OD315 of diluted wines**
6. **Proline**

## Tech Stack
- **Python**: Core programming language.
- **Flask**: Web framework for the application.
- **Scikit-learn**: Machine learning library for model training.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **HTML/CSS**: Frontend design.

## Installation and Usage

### Prerequisites
Ensure you have Python installed.

### Steps
1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   If the model files (`model/wine_cultivar_model.pkl`, `model/scaler.pkl`) are missing or need updating, run:
   ```bash
   python model/model_building.py
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the App**:
   Open your browser and visit: `http://127.0.0.1:5000/`

## File Structure
- `app.py`: Main Flask application file.
- `model/`: Contains the model training script and saved model artifacts.
    - `model_building.py`: Script to train and save the model.
    - `wine_cultivar_model.pkl`: Saved Random Forest model.
    - `scaler.pkl`: Saved Scaler for data preprocessing.
- `static/`: CSS and static assets.
- `templates/`: HTML templates.
- `requirements.txt`: Python dependencies.

## Author
**Onuegbu Udochukwu**
*Wine Cultivar Prediction Project*
