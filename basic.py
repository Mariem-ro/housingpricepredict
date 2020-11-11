from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import pandas as pd
data_preparation = joblib.load(open('dataPreparation.pkl', 'rb'))
final_model = joblib.load(open('RandomForestRegressor.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def man():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def home():
    longitude = request.form.get('Longitude')
    atitude = request.form.get('Latitude')
    housing_median_age = request.form.get('Housing median age')
    totals_rooms = request.form.get('Totals rooms')
    totals_bedrooms = request.form.get('Total bedrooms')
    population = request.form.get('Population')
    households = request.form.get('Householders')
    median_income = request.form.get('Median income')
    ocean_proximity = request.form['toggle_option']
    
    #features extraction
    rooms_per_household = float(totals_rooms)/ float(households)
    bedrooms_per_room = float(totals_bedrooms)/ float(totals_rooms)
    population_per_household = float(totals_rooms)/ float(households)

    features = np.array([longitude, atitude, housing_median_age, totals_rooms, population, households, median_income, ocean_proximity, rooms_per_household, bedrooms_per_room, population_per_household])
    features_df = pd.DataFrame(data=[features], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'population', 'households', 'median_income', 'ocean_proximity', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household'])
    clean_features = data_preparation.transform(features_df)
    prediction = final_model.predict(clean_features)
    return render_template('index2.html', prediction_text='price of the house will be {}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)
    