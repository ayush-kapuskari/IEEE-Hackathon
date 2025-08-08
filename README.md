# IEEE-Hackathon


Solar Power Prediction Project
Overview
This project focuses on predicting solar power production using machine learning techniques, specifically an XGBoost Regressor model. The dataset, derived from /spg.csv, includes meteorological variables such as  radiation,  temperature, and relative humidity, zenith alongside the target variable Power Genration. The goal is to develop a robust model to forecast solar power output based on environmental conditions.
Features

Data Source: kaggle with covering a full year (4213  data points).
Key Variables:
Power Genration: Target variable, representing solar power output (range: 0 to 3056.794100 units, mean: 1134.347313).
Temperature: Ambient temperature in degrees Celsius.




Model: XGBoost Regressor with an R² score of 0.8103 and RMSE of 413.9915 on the test set.

Installation

Clone the repository:git clone "https://github.com/ayush-kapuskari/IEEE-Hackathon/tree/main"
cd IEEE-Hackathon


Set up a Python environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install required packages:pip install numpy pandas scikit-learn xgboost matplotlib seaborn


Ensure Google Colab is set up with runtime type "Python 3" for notebook execution.

Usage

Open the Jupyter Notebook Untitled3.ipynb in Google Colab or locally.
Load the dataset /spg_cleaned.csv.
Run the cells to preprocess data, train the XGBoost model, and evaluate performance.
Adjust hyperparameters or add features as needed (e.g., temporal features like hour of day).

Data Preprocessing

Threshold value: remove the features's whos realtion is above 0.89 andd kept only one

Model Performance

Training Set:Root Mean Squared Error: 167.5232, R² Score: 0.9677
Test Set: Root Mean Squared Error: 413.9915, R² Score: 0.8103
The scatter plot of actual vs. predicted values shows good alignment, with some spread indicating room for improvement.

Future Improvements

Feature Engineering: Incorporate weather data (e.g., Relative Humidity, Zenith for better accuracy.
Hyperparameter Tuning: Optimize XGBoost parameters using grid search.
Error Analysis: Investigate errors during low production periods .
Cross-Validation: Implement time-series cross-validation for robust evaluation.

Contributing
Feel free to fork this repository, submit issues, or create pull requests for enhancements. Suggestions for additional features or data sources are welcome.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Data sourced from kaggle.
Built using Google Colab and Python libraries (NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn).
