from flask import Flask, request, jsonify
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained logistic regression model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request
    data = request.json
    features = [
        data['radius_mean'], data['texture_mean'], data['perimeter_mean'], data['area_mean'],
        data['smoothness_mean'], data['compactness_mean'], data['concavity_mean'], data['concave points_mean'],
        data['symmetry_mean'], data['fractal_dimension_mean'], data['radius_se'], data['texture_se'],
        data['perimeter_se'], data['area_se'], data['smoothness_se'], data['compactness_se'],
        data['concavity_se'], data['concave points_se'], data['symmetry_se'], data['fractal_dimension_se'],
        data['radius_worst'], data['texture_worst'], data['perimeter_worst'], data['area_worst'],
        data['smoothness_worst'], data['compactness_worst'], data['concavity_worst'], data['concave points_worst'],
        data['symmetry_worst'], data['fractal_dimension_worst']
    ]

    # Convert features to numpy array and reshape for prediction
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    
    # Return the prediction result as JSON
    result = 'Malignant' if prediction[0] == 0 else 'Benign'
    return jsonify({'prediction': result})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
