from flask import Flask, jsonify, request
import os
import time
import joblib
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask_cors import CORS

from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import shap

import requests
from io import BytesIO
import random

app = Flask(__name__)
CORS(app)
# Load model
model_url = "https://raw.githubusercontent.com/Pragadesh-45/maritime/main/Ship_S5_XG_Sensor2"
response = requests.get(model_url)
model_ = joblib.load(BytesIO(response.content))

# Create a SHAP explainer
explainer = shap.Explainer(model_)


@app.route('/predict', methods=['POST'])
def predict():
  try:
    # Get input data from the request
    input_data = request.get_json()
    
    # Convert input data to a 1D numpy array
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    # Predict the y (fuel consumption rate, MT/day) values
    y_pred = model_.predict(input_array)
    
    
    # Calculate SHAP values on the input data
    shap_values = explainer.shap_values(input_array)
    
    # Find the most contributing feature
    most_contributing_feature_index = np.argmax(np.abs(shap_values))
    most_contributing_feature_name = list(input_data.keys())[most_contributing_feature_index]
    
    # Build the response
    prediction = {"prediction": float(y_pred * 24/1000),               "most_contributing_feature":most_contributing_feature_name,}
    
    x = prediction
    
    return jsonify(x)

  except Exception as e:
    return jsonify({"error": str(e)})


app.run(host='0.0.0.0', port=81)
