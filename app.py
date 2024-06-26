from flask import Flask, request, render_template, jsonify
import pandas as pd
import logging
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo desde JSON y los pesos desde HDF5
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.weights.h5")
app.logger.debug('Modelo cargado correctamente.')

# Compilar el modelo cargado
loaded_model.compile(loss='mean_squared_error', optimizer='adam')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        enginesize = float(request.form['enginesize'])
        curbweight = float(request.form['curbweight'])
        horsepower = float(request.form['horsepower'])
        highwaympg = float(request.form['highwaympg'])
        carwidth = float(request.form['carwidth'])
        stroke = float(request.form['stroke'])
        peakrpm = float(request.form['peakrpm'])
        carheight = float(request.form['carheight'])
        boreratio = float(request.form['boreratio'])
        symboling = float(request.form['symboling'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[enginesize, curbweight, horsepower, highwaympg, carwidth, stroke, peakrpm, carheight, boreratio, symboling]], 
                               columns=['enginesize', 'curbweight', 'horsepower', 'highwaympg', 'carwidth', 'stroke', 'peakrpm', 'carheight', 'boreratio', 'symboling'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = loaded_model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0][0]}')
        
        # Convertir la predicción a float
        prediction_float = float(prediction[0][0])
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'prediccion': prediction_float})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
