from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load dataset
car = pd.read_csv('quikr_car.csv')
car = car[car['year'].astype(str).str.isnumeric()]
car['year'] = car['year'].astype(int)
car['fuel_type'] = car['fuel_type'].astype(str)
car = car[~car['fuel_type'].isin(['nan', 'NaN', 'None', '', '...'])]

companies = sorted(car['company'].dropna().unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = sorted(car['fuel_type'].dropna().unique())

model = joblib.load(os.path.join('models', 'car_price_model.pkl'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        name = request.form.get('name')
        company = request.form.get('company')
        year = int(request.form.get('year'))
        kms_driven = int(request.form.get('kms_driven'))
        fuel_type = request.form.get('fuel_type')

        input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

    return render_template('index.html',
                           companies=companies,
                           years=years,
                           fuel_types=fuel_types,
                           prediction=prediction)

@app.route('/get_models', methods=['POST'])
def get_models():
    data = request.get_json()
    selected_company = data.get('company')
    if not selected_company:
        return jsonify([])

    df = pd.read_csv('quikr_car.csv')
    df = df[df['Price'] != 'Ask For Price']
    models = sorted(df[df['company'] == selected_company]['name'].unique())
    return jsonify(models)


if __name__ == '__main__':
    app.run(debug=True)