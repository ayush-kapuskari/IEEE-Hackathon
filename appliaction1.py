from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd


application1 = Flask(__name__)
app = application1

XGBOOST_model = pickle.load(open('XGBOOST_MODEl_reduced.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/solar_predication', methods=['POST', 'GET'])
def solar_predication():
    if request.method == 'POST':
        temperature_2_m_above_gnd = float(request.form.get("temperature_2_m_above_gnd",0.0))
        relative_humidity_2_m_above_gnd = float(request.form.get("relative_humidity_2_m_above_gnd",0.0))
        mean_sea_level_pressure_MSL = float(request.form.get("mean_sea_level_pressure_MSL",1013.25))
        
        wind_speed_10_m_above_gnd = float(request.form.get("wind_speed_10_m_above_gnd",5.0))
        
        
        
        
        total_cloud_cover_sfc = float(request.form.get("total_cloud_cover_sfc",0.0))
        shortwave_radiation_backwards_sfc = float(request.form.get("shortwave_radiation_backwards_sfc",0.0))
        
        
        
        angle_of_incidence_sfc = float(request.form.get("angle_of_incidence_sfc",0.0))
        zenith_sfc = float(request.form.get("zenith_sfc",45.0))
        azimuth_sfc = float(request.form.get("azimuth_sfc",0.0))

        new_data = [[angle_of_incidence_sfc,zenith_sfc,azimuth_sfc,shortwave_radiation_backwards_sfc,total_cloud_cover_sfc,mean_sea_level_pressure_MSL,relative_humidity_2_m_above_gnd,
                         temperature_2_m_above_gnd,wind_speed_10_m_above_gnd
                         ]]
        x_train_reduced_columns = ['angle_of_incidence', 'zenith', 'azimuth', 'shortwave_radiation_backwards_sfc', 'total_cloud_cover_sfc',
                           'mean_sea_level_pressure_MSL', 'relative_humidity_2_m_above_gnd', 'temperature_2_m_above_gnd',
                           'wind_speed_10_m_above_gnd']
        
        new_df = pd.DataFrame(new_data, columns=x_train_reduced_columns)
        prediction = XGBOOST_model.predict(new_df)

        

        return render_template('solar_predication.html', results_score=prediction[0])
    else:
        return render_template('solar_predication.html', results_score='please enter the data first')

if __name__ == "__main__":
    app.run(debug=True, port=5002)