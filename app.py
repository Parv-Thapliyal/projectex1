import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
@app.route('/')
def home():
    return render_template('index.html')

...

if __name__ == '__main__':
# This is a crucial line for running in a non-GUI environment like a server
matplotlib.use('Agg') 

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for local development
CORS(app) 

# Load the trained model and scaler
try:
    model = joblib.load('diabetes_gb_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    df = pd.read_csv('diabetes.csv')
    print("Model, scaler, and dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Required files (model.pkl, scaler.pkl, diabetes.csv) not found. Please run the training script first.")
    model = None
    scaler = None
    df = None

# A helper function to create a plot and return it as a base64 string
def create_plot_image(plot_func):
    """
    Creates a plot using the provided function and returns a base64 encoded string of the PNG image.
    """
    if df is None:
        return ""
    
    buf = io.BytesIO()
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    
    plot_func()
    
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/eda_plots', methods=['GET'])
def get_eda_plots():
    """
    Generates and returns all static EDA plots as a JSON object of base64-encoded images.
    """
    if df is None:
        return jsonify({"error": "Dataset not loaded."}), 500

    plots = {}
    
    # Plot 1: Outcome Distribution
    plots['outcome_distribution'] = create_plot_image(
        lambda: sns.countplot(x='Outcome', data=df).set_title('Distribution of Outcome (0: Non-Diabetic, 1: Diabetic)')
    )
    
    # Plot 2: Correlation Heatmap
    plots['correlation_heatmap'] = create_plot_image(
        lambda: sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f').set_title('Correlation Heatmap')
    )
    
    # Plot 3: Glucose Levels by Outcome
    plots['glucose_boxplot'] = create_plot_image(
        lambda: sns.boxplot(x='Outcome', y='Glucose', data=df).set_title('Glucose Levels by Outcome')
    )
    
    # Plot 4: BMI by Outcome
    plots['bmi_boxplot'] = create_plot_image(
        lambda: sns.boxplot(x='Outcome', y='BMI', data=df).set_title('BMI by Outcome')
    )
    
    # Plot 5: Age vs. Glucose Scatterplot
    plots['age_glucose_scatterplot'] = create_plot_image(
        lambda: sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=df, alpha=0.7).set_title('Age vs. Glucose by Outcome')
    )
    
    # Plot 6: Feature Histograms
    # This one is a bit special, using df.hist()
    def plot_histograms():
        df.hist(bins=20, figsize=(15, 10))
        plt.suptitle('Histograms of All Features', y=1.02)
    plots['feature_histograms'] = create_plot_image(plot_histograms)
    
    return jsonify(plots)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives new data, makes a prediction, and returns the result along with a dynamic plot.
    """
    if model is None or scaler is None or df is None:
        return jsonify({"error": "Model, scaler, or dataset not loaded."}), 500
    
    data = request.get_json(force=True)
    
    # Create a DataFrame from the new data
    new_data = pd.DataFrame([data])
    
    # Scale the new data
    scaled_data = scaler.transform(new_data)
    
    # Make a prediction
    prediction = model.predict(scaled_data)
    
    # --- Dynamic Plot Generation ---
    # Create the dynamic plot with the new data point
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=df, alpha=0.7)
    
    # Add the user's data point
    user_color = 'black' if prediction[0] == 0 else 'red'
    plt.scatter(data['Age'], data['Glucose'], color=user_color, s=200, marker='*', edgecolor='white', linewidth=2, label='Your Input')
    
    plt.title('Age vs. Glucose with Your Input')
    plt.xlabel('Age')
    plt.ylabel('Glucose')
    plt.legend()
    
    # Save plot to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Encode plot as base64 string
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return jsonify({
        "prediction": int(prediction[0]),
        "plot_image": plot_base64
    })


if __name__ == '__main__':
    app.run(debug=True)

