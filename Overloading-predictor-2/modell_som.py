import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

def train_model():
# Läs in dataset med korrekta filnamn
    temp_data = pd.read_csv("Sheet1+ (Flera anslutningar)_Lufttemp.csv")
    passenger_data = pd.read_csv("Sheet1+ (Flera anslutningar)_Passagerare.csv")

    # Förbehandla temperaturdata
    temp_data['date'] = pd.to_datetime(temp_data["Datum (Sheet11)"]).dt.date
    temp_data['temperature'] = temp_data['Lufttemperatur']
            
    # Förbehandla passagerardata med tid
    passenger_data['datetime'] = pd.to_datetime(passenger_data['Tidpunkt'])
    passenger_data['hour'] = passenger_data['datetime'].dt.hour
    passenger_data['date'] = passenger_data['datetime'].dt.date
    passenger_data['passengers'] = pd.to_numeric(passenger_data['Totalt antal påstigande'], errors='coerce')
            
    # Filtrera endast UT-resor för passagerardata
    passenger_data = passenger_data[passenger_data['Väg'] == 'UT']
            
    # Konvertera date-kolumnerna till samma typ (datetime64[ns])
    temp_data['date'] = pd.to_datetime(temp_data['date'])
    passenger_data['date'] = pd.to_datetime(passenger_data['date'])
            
    # Aggregera till timvisa värden per dag
    hourly_data = pd.merge(
        temp_data.groupby('date')['temperature'].mean().reset_index(),
        passenger_data.groupby(['date', 'hour'])['passengers'].mean().reset_index(),
        on='date'
        )

    # Hantera eventuella saknade värden
    hourly_data = hourly_data.dropna()

    X = hourly_data[['temperature', 'hour']]
    y = hourly_data['passengers']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #Lek runt med parametrar i RandomForrest för att förbättra modelen, kolla på hemsidan för vilka parametrar som finns
    model = RandomForestRegressor(n_estimators=20000,
                                max_depth=10,
                                min_samples_split=5)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"MSE: {mse:.2f}")

    return model  # ← så du kan använda modellen senare

    #Spottar ut Mean Squared Error av modellen, kan tänka på det som 
    #hur mycket varierar modellens fel. Desto mindre desto bättre.
    #print(f"MSE: {np.sqrt(mean_squared_error(y_test, pred))}")