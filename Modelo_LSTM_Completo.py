# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:12:43 2024

@author: USUARIO
"""

import seaborn as sns
from math import sqrt
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import datetime

class Model_LSTM:
    def __init__(self, variables, start_date, look_back):
        self.variables = variables
        self.start_date = start_date
        self.end_date = datetime.date.today()  # Obtener la fecha actual
        self.look_back = look_back
        self.model = None
        
    def load_data(self):
        end_date_str = self.end_date.strftime('%Y-%m-%d')
        data_history = yf.download(self.variables, start=self.start_date, end=end_date_str)['Adj Close'].dropna()
        return data_history

    def preprocess_data(self, data_history, inicio):
        data_scaled = data_history[inicio:]
        data_scaled = data_scaled.values
        data_scaled = data_scaled.reshape((len(data_scaled), 8))
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled_total = scaler.fit_transform(data_scaled)
        return data_scaled_total
    
    def create_dataset(self, dataset, look_back):
        data_x = []
        data_y = []

        for i in range(look_back, len(dataset)):
            data_x.append(dataset[i - look_back:i, :])  # Incluir todas las características en la entrada
            data_y.append([dataset[i, 1]])  # Formato bidimensional para un solo valor de salida
    
        return np.array(data_x), np.array(data_y)
    
    def split_data(self, X, y, train_ratio=0.8):
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test
        

    def build_Model(self, units, dropout_rate, dense_units, learning_rate, activation, back_size=None):
        LSTM_model = Sequential()
        LSTM_model.add(LSTM(units, 
                            activation=activation, 
                            dropout=dropout_rate, 
                            recurrent_dropout=dropout_rate,
                            input_shape=(self.look_back, 8)))
        LSTM_model.add(Dropout(dropout_rate))
        LSTM_model.add(Dense(dense_units, activation=activation))
        LSTM_model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        LSTM_model.compile(loss='mean_squared_error', optimizer=optimizer)
        return LSTM_model

    def train_model(self, X_train, y_train, epochs, batch_size, validation_data):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=2)
        return history

    def plot_training_history(self, history):
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_future(self, n_days):
        # Obtener los últimos datos disponibles
        data_predict = model.load_data()[-self.look_back:]
        data_scalada = model.preprocess_data(data_predict, inicio=0)
        
        # Crear un conjunto de datos con los últimos datos disponibles
        X_last = data_scalada[-self.look_back:].reshape(1, self.look_back, 8)
        
        # Realizar predicciones para los próximos n_days días
        new_predictions = []
        for _ in range(n_days):
            pred = self.model.predict(X_last)[0][0]  # Predicción del EUR/USD
            new_predictions.append(pred)
            
            # Actualizar el conjunto de datos con la última predicción del EUR/USD
            X_last = np.roll(X_last, -1, axis=1)
            X_last[0][-1][1] = pred  # Actualizar la segunda columna (EUR/USD)
            
            # Mantener los últimos valores conocidos de las demás variables solo para la predicción actual
            X_last[0, -1, [0, 2, 3, 4, 5, 6, 7]] = X_last[0, -2, [0, 2, 3, 4, 5, 6, 7]]
        
        # Desescalar las predicciones
        predic_scaled = MinMaxScaler(feature_range=(0, 1))
        predic_scaled.fit(data_predict['EURUSD=X'].values.reshape(-1, 1))
        predicciones_reescaladas = predic_scaled.inverse_transform(np.array(new_predictions).reshape(-1, 1))
        
        #print(X_last)
        return predicciones_reescaladas
    
    def plot_predictions(self):
   
        # Graficar datos reales y predicciones
        plt.figure(figsize=(10, 6))
        plt.plot(scaled_ytest, color='blue', label='Real EUR/USD Price', alpha=0.8)
        plt.plot(pred_rescaled, color='red', label='Predicted EUR/USD Price', alpha=0.8)
        plt.title('EUR/USD Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('EUR/USD stock price')
        plt.legend()
        plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para mayor legibilidad
        plt.tight_layout()
        plt.show()
        
        # Concatenar los valores reales y las predicciones futuras
        combined_data = list(scaled_ytest) + list(future_predictions)
    
        # Graficar datos reales y predicciones futuras
        plt.figure(figsize=(10, 6))
        plt.plot(combined_data, color='red', label='Real EUR/USD Price & Future Prediction', alpha=0.8)
        plt.plot(scaled_ytest, color='blue', label='Real EUR/USD Price', alpha=0.8)
        plt.title('EUR/USD Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('EUR/USD stock price')
        plt.legend()
        plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para mayor legibilidad
        plt.tight_layout()
        plt.show()
        
# main        
        
model = Model_LSTM(variables=['EURUSD=X', '^GSPC', '^DJI', '^IXIC', '^RUT', 'CL=F', '^FTSE', '^GDAXI'],
                   start_date='2017-01-01',
                   look_back=60)


data = model.load_data()
data_scaled = model.preprocess_data(data, inicio=0)
X, y = model.create_dataset(data_scaled, look_back=60)

# Dividir los datos en conjuntos de entrenamiento y prueba de manera determinista
X_train, X_test, y_train, y_test = model.split_data(X, y, train_ratio=0.8)

# Definir hiperparámetros manualmente
units = 180
dropout_rate = 0.05047822526216488
dense_units = 60
learning_rate = 0.018900608121585874
activation = 'sigmoid'

# Construir el modelo con los hiperparámetros definidos manualmente
model.model = model.build_Model(units, dropout_rate, dense_units, learning_rate, activation)

# Entrenar el modelo
history = model.train_model(X_train, y_train, epochs=160, batch_size=26, validation_data=(X_test, y_test))
model.plot_training_history(history)

# Realizar predicciones
predictions = model.predict(X_test)

# Crear una nueva instancia del MinMaxScaler
scaler_pred = MinMaxScaler(feature_range=(0, 1))

# Ajustar el scaler con los datos de entrenamiento
scaler_pred.fit(data['EURUSD=X'].values.reshape(-1, 1))

# Desescalar las predicciones a su forma original
pred_rescaled = scaler_pred.inverse_transform(predictions)


scaled_ytest = scaler_pred.inverse_transform(y_test)


# Calcular el RMSE
rmse = np.sqrt(mean_squared_error(scaled_ytest, pred_rescaled))
print('Test RMSE: %.3f' % rmse)

# Realizar predicciones para los próximos 5 días
future_predictions = model.predict_future(5)
print("Predicciones para el proximo dia:")
print(future_predictions)

model.plot_predictions()
