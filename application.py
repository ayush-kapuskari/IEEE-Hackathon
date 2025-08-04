from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

random_forest_model = pickle.load(open('random_forest_model1.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/solar_predication', methods=['POST', 'GET'])
def solar_predication():
    if request.method == 'POST':
        temperature_2_m_above_gnd = float(request.form.get("temperature_2_m_above_gnd"))
        relative_humidity_2_m_above_gnd = float(request.form.get("relative_humidity_2_m_above_gnd"))
        mean_sea_level_pressure_MSL = float(request.form.get("mean_sea_level_pressure_MSL"))
        total_precipitation_sfc = float(request.form.get("total_precipitation_sfc"))
        wind_speed_10_m_above_gnd = float(request.form.get("wind_speed_10_m_above_gnd"))
        snowfall_amount_sfc = float(request.form.get("snowfall_amount_sfc"))
        total_cloud_cover_sfc = float(request.form.get("total_cloud_cover_sfc"))
        high_cloud_cover_high_cld_lay = float(request.form.get("high_cloud_cover_high_cld_lay"))
        medium_cloud_cover_medium_cld_lay = float(request.form.get("medium_cloud_cover_medium_cld_lay"))
        low_cloud_cover_low_cld_lay = float(request.form.get("low_cloud_cover_low_cld_lay"))
        shortwave_radiation_flux_sfc = float(request.form.get("shortwave_radiation_flux_sfc"))
        wind_direction_10_m_above_gnd = float(request.form.get("wind_direction_10_m_above_gnd"))
        wind_direction_80_m_above_gnd = float(request.form.get("wind_direction_80_m_above_gnd"))
        win_gust_speed_10_m_above_gnd = float(request.form.get("win_gust_speed_10_m_above_gnd"))
        angle_of_incidence = float(request.form.get("angle_of_incidence_sfc"))
        zenith = float(request.form.get("zenith_sfc"))
        azimuth = float(request.form.get("azimuth_sfc"))

        new_standard_data = scaler.transform([[temperature_2_m_above_gnd, relative_humidity_2_m_above_gnd, mean_sea_level_pressure_MSL, total_precipitation_sfc, wind_speed_10_m_above_gnd, snowfall_amount_sfc, total_cloud_cover_sfc, high_cloud_cover_high_cld_lay, medium_cloud_cover_medium_cld_lay, low_cloud_cover_low_cld_lay, shortwave_radiation_flux_sfc, wind_direction_10_m_above_gnd, wind_direction_80_m_above_gnd, win_gust_speed_10_m_above_gnd, angle_of_incidence, zenith, azimuth]])
        prediction = random_forest_model.predict(new_standard_data)

        return render_template('solar_predication.html', results_score=prediction[0])
    else:
        return render_template('solar_predication.html', results_score='please enter the data first')

if __name__ == "__main__":
    app.run(debug=True, port=5002)