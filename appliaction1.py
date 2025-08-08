from flask import Flask, render_template, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np
import json

loaded_model = xgb.XGBRegressor()
loaded_model.load_model("xgboost_regressor_reduced.json")

try:
    with open("feature_names.json", "r") as f:
        x_train_reduced_columns = json.load(f)
except FileNotFoundError:
    
    x_train_reduced_columns = [
        'angle_of_incidence', 
        'zenith', 
        'azimuth', 
        'shortwave_radiation_backwards_sfc', 
        'total_cloud_cover_sfc', 
        'mean_sea_level_pressure_MSL', 
        'relative_humidity_2_m_above_gnd', 
        'temperature_2_m_above_gnd', 
        'wind_speed_10_m_above_gnd'
    ]

application1 = Flask(__name__)
app = application1



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/solar_predication', methods=['POST', 'GET'])
def solar_predication():
    if request.method == 'POST':
        
        try:
            input_data = {
                'angle_of_incidence': float(request.form.get("angle_of_incidence", 0.0)),
                'zenith': float(request.form.get("zenith_sfc", 45.0)),
                'azimuth': float(request.form.get("azimuth_sfc", 0.0)),
                'shortwave_radiation_backwards_sfc': float(request.form.get("shortwave_radiation_backwards_sfc", 0.0)),
                'total_cloud_cover_sfc': float(request.form.get("total_cloud_cover_sfc", 0.0)),
                'mean_sea_level_pressure_MSL': float(request.form.get("mean_sea_level_pressure_MSL", 1013.25)),
                'relative_humidity_2_m_above_gnd': float(request.form.get("relative_humidity_2_m_above_gnd", 0.0)),
                'temperature_2_m_above_gnd': float(request.form.get("temperature_2_m_above_gnd", 0.0)),
                'wind_speed_10_m_above_gnd': float(request.form.get("wind_speed_10_m_above_gnd", 5.0))
            }
        except (ValueError, TypeError) as e:
            return render_template('solar_predication.html', results_score=f"Error processing input data: {e}. Please enter valid numbers.")

        new_df = pd.DataFrame([input_data], columns=x_train_reduced_columns)
        
        print(input_data)
       
        print("Input DataFrame before prediction:")
        print("Data types of input DataFrame:")
        for col in new_df.columns:
            print(f"{col}: {new_df[col].values[0]}")
        
        print(new_df.dtypes)
      
        
        prediction = loaded_model.predict(new_df)
        print("Prediction:", prediction)
        return render_template('solar_predication.html', results_score=f"Predicted Solar Power: {prediction[0]:.2f}")
    else:
        return render_template('solar_predication.html', results_score='Please enter the data first')

if __name__ == "__main__":
    app.run(debug=True, port=5002)