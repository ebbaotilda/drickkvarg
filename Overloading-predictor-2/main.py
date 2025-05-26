# Uppdaterad till Streamlit med VrångGo-namn, appförklaring, utan karta och timvis väderdata i avgångstider
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import traceback
from modell_som import train_model

# ResRobot API-nyckel och stopp-ID
access_id = "25add2dc-d84e-4119-89f3-091d4c78c3d5"
saltholmen_id = "740001206"
vrango_id = "740001382"

# Lista över Sveriges röda dagar 2025 (manuellt definierad)
SWEDISH_PUBLIC_HOLIDAYS_2025 = [
    datetime(2025, 1, 1),   # Nyårsdagen
    datetime(2025, 1, 6),   # Trettondedag jul
    datetime(2025, 4, 18),  # Långfredagen
    datetime(2025, 4, 20),  # Påskdagen
    datetime(2025, 4, 21),  # Annandag påsk
    datetime(2025, 5, 1),   # Första maj
    datetime(2025, 5, 29),  # Kristi himmelsfärdsdag
    datetime(2025, 6, 6),   # Sveriges nationaldag
    datetime(2025, 6, 21),  # Midsommardagen
    datetime(2025, 11, 1),  # Alla helgons dag
    datetime(2025, 12, 25), # Juldagen
    datetime(2025, 12, 26), # Annandag jul
]

# Lista över svenska veckodagsnamn
SWEDISH_WEEKDAYS = ["Måndag", "Tisdag", "Onsdag", "Torsdag", "Fredag", "Lördag", "Söndag"]

def is_public_holiday(date):
    return date.date() in [d.date() for d in SWEDISH_PUBLIC_HOLIDAYS_2025]

class PassengerPredictor:
    def __init__(self):
        self.model = None
        self.le_direction = LabelEncoder()
        try:
            self.load_data()
            self.train_model()
        except Exception as e:
            st.error(f"Fel vid initiering av PassengerPredictor: {str(e)}")
            st.error(traceback.format_exc())
    
    def load_data(self):
        try:
            # Kontrollera att filerna finns
            for file in ["Sheet1+ (Flera anslutningar)_Lufttemp.csv", "Sheet1+ (Flera anslutningar)_Passagerare.csv", "tabx.xlsx"]:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Filen {file} saknas i den aktuella katalogen.")

            self.temp_data = pd.read_csv("Sheet1+ (Flera anslutningar)_Lufttemp.csv")
            self.passenger_data_old = pd.read_csv("Sheet1+ (Flera anslutningar)_Passagerare.csv")
            self.passenger_data_new = pd.read_excel("tabx.xlsx")
            
            self.temp_data["Timestamp"] = pd.to_datetime(self.temp_data["Tidpunkt timvis (Sheet11)"], format="%m/%d/%Y %I:%M:%S %p")
            self.temp_data["date"] = self.temp_data["Timestamp"].dt.date
            self.temp_data["temperature"] = self.temp_data["Lufttemperatur"]
            
            self.passenger_data_old["datetime"] = pd.to_datetime(self.passenger_data_old["Tidpunkt"], format="%m/%d/%Y %I:%M:%S %p")
            self.passenger_data_old["hour"] = self.passenger_data_old["datetime"].dt.hour
            self.passenger_data_old["date"] = self.passenger_data_old["datetime"].dt.date
            self.passenger_data_old["passengers"] = pd.to_numeric(self.passenger_data_old["Totalt antal påstigande"], errors="coerce")
            
            if not pd.api.types.is_datetime64_any_dtype(self.passenger_data_new["Datum"]):
                self.passenger_data_new["Datum"] = pd.to_datetime(self.passenger_data_new["Datum"], unit="D", origin="1899-12-30")
            else:
                self.passenger_data_new["Datum"] = pd.to_datetime(self.passenger_data_new["Datum"])
            
            self.passenger_data_new["Tid"] = self.passenger_data_new["Tid"].astype(str).str.replace(".", ":")
            self.passenger_data_new["datetime"] = pd.to_datetime(
                self.passenger_data_new["Datum"].dt.strftime("%Y-%m-%d") + " " + self.passenger_data_new["Tid"]
            )
            self.passenger_data_new["hour"] = self.passenger_data_new["datetime"].dt.hour
            self.passenger_data_new["date"] = self.passenger_data_new["datetime"].dt.date
            self.passenger_data_new["passengers"] = pd.to_numeric(self.passenger_data_new["Totalt antal påstigande"], errors="coerce")
            
            passenger_data_combined = pd.concat(
                [
                    self.passenger_data_old[["datetime", "hour", "date", "passengers", "Väg"]],
                    self.passenger_data_new[["datetime", "hour", "date", "passengers", "Väg"]]
                ],
                ignore_index=True
            )
            
            self.temp_data["date"] = pd.to_datetime(self.temp_data["date"])
            passenger_data_combined["date"] = pd.to_datetime(passenger_data_combined["date"])
            
            self.hourly_data = pd.merge(
                self.temp_data.groupby("date")["temperature"].mean().reset_index(),
                passenger_data_combined.groupby(["date", "hour", "Väg"])["passengers"].mean().reset_index(),
                on="date"
            )

            self.hourly_data["overload"] = (self.hourly_data["passengers"] > 0.8*163).astype(int)
            
            self.hourly_data = self.hourly_data.dropna()
            if self.hourly_data.empty:
                raise ValueError("Ingen data tillgänglig efter bearbetning (hourly_data är tom).")
            
            self.hourly_data["Direction_Encoded"] = self.le_direction.fit_transform(self.hourly_data["Väg"])
            
            self.hourly_data["Is_Public_Holiday"] = self.hourly_data["date"].apply(
                lambda x: 1 if is_public_holiday(x) else 0
            )
            self.hourly_data["Is_Weekend"] = self.hourly_data["date"].apply(
                lambda x: 1 if x.weekday() >= 5 else 0
            )
        except Exception as e:
            st.error(f"Fel vid laddning av data: {str(e)}")
            st.error(traceback.format_exc())
            raise
    
    def train_model(self):
        try:
            X = self.hourly_data[["temperature", "hour", "Direction_Encoded", "Is_Public_Holiday", "Is_Weekend"]]
            y = self.hourly_data["overload"]
            
            if X.empty or y.empty:
                raise ValueError("Ingen data tillgänglig för träning (X eller y är tom).")
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
        except Exception as e:
            st.error(f"Fel vid träning av modellen: {str(e)}")
            st.error(traceback.format_exc())
            raise
    
    def predict(self, temperature, hour, direction, is_public_holiday, is_weekend):
        try:
            direction_encoded = self.le_direction.transform([direction])[0]
            input_data = pd.DataFrame(
                [[temperature, hour, direction_encoded, is_public_holiday, is_weekend]],
                columns=["temperature", "hour", "Direction_Encoded", "Is_Public_Holiday", "Is_Weekend"]
            )
            return self.model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Fel vid prognos: {str(e)}")
            st.error(traceback.format_exc())
            return 0

def get_weather_forecast(date, selected_time):
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=57.7&longitude=11.97&hourly=temperature_2m,precipitation_probability,precipitation,rain,uv_index&timezone=Europe%2FBerlin&forecast_days=16"
        response = requests.get(url)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        forecast_times = data["hourly"]["time"]
        temperatures = data["hourly"]["temperature_2m"]
        uv_indices = data["hourly"]["uv_index"]
        precipitation_probs = data["hourly"]["precipitation_probability"]
        precipitations = data["hourly"]["precipitation"]
        rains = data["hourly"]["rain"]

        # Omvandla forecast_times till datetime-objekt
        forecast_datetimes = [datetime.strptime(time_str, "%Y-%m-%dT%H:%M") for time_str in forecast_times]

        # Skapa target_datetime baserat på datum och vald tid
        target_datetime = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {selected_time}", "%Y-%m-%d %H:%M")
        target_datetime = target_datetime.replace(minute=0, second=0, microsecond=0)  # Runda till närmaste timme

        # Hitta närmaste timme i väderdata
        closest_index = min(range(len(forecast_datetimes)), key=lambda i: abs(forecast_datetimes[i] - target_datetime))

        # Hämta väderdata för den valda timmen
        weather_data = {
            "temperature": temperatures[closest_index],
            "uv_index": uv_indices[closest_index],
            "precipitation_probability": precipitation_probs[closest_index],
            "precipitation": precipitations[closest_index],
            "rain": rains[closest_index],
            "time": forecast_datetimes[closest_index].strftime("%Y-%m-%d %H:%M")
        }
        return weather_data
    except Exception as e:
        st.error(f"Fel vid hämtning av väderprognos: {str(e)}")
        st.error(traceback.format_exc())
        return None

def get_trips(origin_id, dest_id, date, time=None):
    try:
        url = "https://api.resrobot.se/v2.1/trip"
        results = set()
        
        params = {
            "format": "json",
            "originId": origin_id,
            "destId": dest_id,
            "date": date,
            "numF": 3,
            "passlist": 1,
            "accessId": access_id
        }
        
        if time:
            params["time"] = time
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        trips = data.get("Trip", [])
        for trip in trips:
            for leg in trip.get("LegList", {}).get("Leg", []):
                if leg["type"] == "JNY":
                    product = leg.get("Product", [{}])[0]
                    line_number = product.get("num", "N/A").split(" - ")[0]
                    if "281" in line_number:
                        departure = f"{leg['Origin']['date']} {leg['Origin']['time']}"
                        results.add((
                            departure,
                            line_number,
                            leg.get("direction", "N/A"),
                            f"{leg['Origin']['date']} {leg['Origin']['time']}",
                            f"{leg['Destination']['date']} {leg['Destination']['time']}"
                        ))
        results = [
            {
                "line": line,
                "direction": direction,
                "departure": departure,
                "arrival": arrival
            }
            for departure, line, direction, departure, arrival in sorted(results)
        ]
        return results if results else [{"error": "Inga resor hittades för den valda dagen. 🚤"}]
    except requests.exceptions.HTTPError as e:
        st.error(f"Fel vid hämtning av avgångar: {str(e)}")
        st.error(traceback.format_exc())
        return [{"error": str(e)}]
    except Exception as e:
        st.error(f"Fel vid hämtning av avgångar: {str(e)}")
        st.error(traceback.format_exc())
        return [{"error": str(e)}]

def get_occupancy_level(passengers):
    if passengers == 1:
        st.badge("HÖG RISK", color="violet")
    else:
        st.badge("LÅG RISK", color="green")

st.set_page_config(
    page_title="VrångGo - Prognos och Tidtabell",
    page_icon="⛴️",
    layout="wide"
    )

def main():
    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 0.2, 1, 0.5],
                            gap="large",
    )

    with col2:
        with st.container():
            st.title("VrångGo ⛴️")

            st.markdown("""
                        <link href="https://fonts.googleapis.com/css2?family=Nunito&display=swap" rel="stylesheet">
            <style>
            .info-wrapper {
                position: relative;
                display: inline-block;
                font-family: 'Nunito', sans-serif;
                font-weight: normal;
                font-size: 20px;
            }
            .info-text {
                display: inline;
            }
            .highlight {
                font-style: normal;
            }
            .info-icon {
                display: inline-block;
                background-color: white;
                color: black;
                border: 1.5px solid black;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                text-align: center;
                font-family: serif;
                font-size: 14px;
                line-height: 20px;
                cursor: default;
                margin-left: 8px;
                user-select: none;
            }
            .tooltip {
                display: none;
                position: absolute;
                background-color: #f9f9f9;
                color: #333;
                padding: 10px;
                border-radius: 8px;
                width: 250px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
                font-size: 14px;
                font-family: sans-serif;
                top: 50%;
                left: 100%;
                margin-left: 10px;
                transform: translateY(-50%);
                z-index: 999;
            }
            .tooltip strong {
                font-weight: 600;
                display: block;
                margin-bottom: 3px;
            }
        .tooltip p {
                font-weight: 300;
                display: block;
                margin: 0;
            }
            .info-wrapper:hover .tooltip {
                display: block;
            }
        </style>

        <div class="info-wrapper">
            <span class="info-text">Besök skärgården - <span class="highlight">i lugn och ro</span></span>
            <div class="info-icon">i</div>
            <div class="tooltip">
                <strong>Vad gör appen?</strong><br>
                <p>Ger en prognos för passargerartryck på resor med linje 281 mellan Saltholmen och Vrångö, så att du kan planera din resa bättre.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True) #lägger till utrymme mellan textraderna
        st.markdown("<br>", unsafe_allow_html=True)

        predictor = PassengerPredictor()
        current_time = datetime.now()
        default_date = current_time.date()
        default_time = current_time.strftime("%H:%M")

        # Välj riktning
        direction_options = ["Saltholmen till Vrångö", "Vrångö till Saltholmen"]
        direction_display = st.selectbox("🧭  Välj riktning", options=direction_options, index=0)
        selected_initial_direction_label = direction_display
        selected_initial_direction = "UT" if selected_initial_direction_label == "Saltholmen till Vrångö" else "IN"

        # Tur och retur-alternativ
        show_return_trip = st.checkbox("Visa även returresa (tur och retur)")

        # Datumval med st.date_input
        selected_date = st.date_input(
            "📅  Välj datum",
            value=default_date,
            min_value=default_date,
            max_value=default_date + timedelta(days=15)
        )
        selected_date_str = selected_date.strftime("%Y-%m-%d")

        hours = [f"{h:02d}:00" for h in range(24)]
        time_display = st.selectbox("🕒  Välj tid (valfritt, för närmaste avgångar för första resan)", options=hours, index=int(default_time.split(":")[0]))
        selected_time_str = time_display

        selected_return_time_str = None
        if show_return_trip:
            return_time_display = st.selectbox("🕒  Välj tid för returresa (valfritt)", options=hours, index=int(default_time.split(":")[0]))
            selected_return_time_str = return_time_display

        if st.button("Hämta avgångar och prognoser"):
            with st.spinner("⏳ Hämtar avgångar och prognoser..."):
                try:
                    date_obj = datetime.strptime(selected_date_str, "%Y-%m-%d")
                    if is_public_holiday(date_obj):
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.warning("⚠️  Detta är en röd dag – fler passagerare kan resa!")

                    def display_trip_info(trips_data, direction_code, section_title, date_obj_param):
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"<h5 style='font-size:16px; margin-bottom:10px;'>{section_title}</h5>", unsafe_allow_html=True)

                        if not trips_data or (isinstance(trips_data, list) and trips_data and "error" in trips_data[0]):
                            error_message = trips_data[0]['error'] if (isinstance(trips_data, list) and trips_data and "error" in trips_data[0]) else "Inga resor hittades."
                            st.error(f"❌ {error_message}")
                            return

                        for trip in trips_data:
                            departure_time_dt = datetime.strptime(trip["departure"], "%Y-%m-%d %H:%M:%S")
                            departure_hour = departure_time_dt.strftime("%H:%M")
                            selected_hour = departure_time_dt.hour
                            is_holiday_val = 1 if is_public_holiday(date_obj_param) else 0
                            is_weekend_val = 1 if date_obj_param.weekday() >= 5 else 0

                            weather_data = get_weather_forecast(date_obj_param, departure_hour)
                            if weather_data is None:
                                st.error(f"❌ Kunde inte hämta väderprognos för avgångstiden {departure_hour}. Open-Meteo API stöder bara prognoser upp till 16 dagar framåt. ☁️")
                                return

                            temperature = weather_data["temperature"]
                            predicted_passengers = int(predictor.predict(
                                temperature, selected_hour, direction_code, is_holiday_val, is_weekend_val
                            ))

                            st.markdown("<br>", unsafe_allow_html=True)

                            with st.container(height=350, border=None):
                                get_occupancy_level(predicted_passengers)
                                #st.badge(f"Passagerare: {predicted_passengers}", color="blue")
                                st.markdown(f"**Linje: {trip['line']}**")
                                st.markdown(f"Avgång: {trip['departure']}")
                                st.markdown(f"Ankomst: {trip['arrival']}")
                                
                                st.badge(f"{weather_data['temperature']:.1f}°C", icon="🌤", color="gray")
                                st.badge(f"{weather_data['precipitation']} mm", icon="💧", color="gray")


                    origin_id_initial = saltholmen_id if selected_initial_direction == "UT" else vrango_id
                    dest_id_initial = vrango_id if selected_initial_direction == "UT" else saltholmen_id
                    initial_trips = get_trips(origin_id_initial, dest_id_initial, selected_date_str, selected_time_str)

                    if show_return_trip:
                        col1, col2 = st.columns(2)
                        with col1:
                            display_trip_info(initial_trips, selected_initial_direction, f"Resa: {selected_initial_direction_label.split('⛴️')[0].strip()}", date_obj)

                        with col2:
                            return_direction = "IN" if selected_initial_direction == "UT" else "UT"
                            return_direction_label = "Vrångö till Saltholmen" if selected_initial_direction == "UT" else "Saltholmen till Vrångö"
                            origin_id_return = vrango_id if selected_initial_direction == "UT" else saltholmen_id
                            dest_id_return = saltholmen_id if selected_initial_direction == "UT" else vrango_id
                            return_trips = get_trips(origin_id_return, dest_id_return, selected_date_str, selected_return_time_str)
                            display_trip_info(return_trips, return_direction, f"Returresa: {return_direction_label}", date_obj)
                    else:
                        display_trip_info(initial_trips, selected_initial_direction, f"Resa: {selected_initial_direction_label.split('⛴️')[0].strip()}", date_obj)

                except Exception as e:
                    st.error(f"Fel vid hämtning av avgångar och prognoser: {str(e)}")
                    st.error(traceback.format_exc())

    with col4:
        col1, col2, col3 = st.columns(3)
        with col3:
            with st.popover(label=":material/favorite:"):
                st.markdown("Dina sparade turer")

        st.markdown("<br>", unsafe_allow_html=True)

        df = pd.read_csv("platser2.csv", sep=';')

    # Bygg en karta
        # Omvandla kolumner till float och ersätt ev. kommatecken
        # Omvandla kolumner till float och ersätt ev. kommatecken
        df["lat"] = df["lat"].astype(str).str.replace(",", ".").astype(float)
        df["lon"] = df["lon"].astype(str).str.replace(",", ".").astype(float)

        # ScatterplotLayer – rosa prickar
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[lon, lat]',
            get_fill_color= [229, 114, 114],
            get_radius=200,
            pickable=True,
        )

        # LineLayer – linje mellan de två första punkterna (om minst två finns), funkar inte, kan vara för att koordinaterna är för nära varandra eventuellt
        line_layer = None
        if len(df) >= 2 and (df.iloc[0]["lat"] != df.iloc[1]["lat"] or df.iloc[0]["lon"] != df.iloc[1]["lon"]):
            line_data = pd.DataFrame([{
                "start": [df.iloc[0]["lon"], df.iloc[0]["lat"]],
                "end": [df.iloc[1]["lon"], df.iloc[1]["lat"]]
            }])
                
            line_layer = pdk.Layer(
                "LineLayer",
                data=line_data,
                get_source_position="start",
                get_target_position="end",
                get_width=4,
                get_color=[0, 0, 0],  # svart linje
                pickable=False
            )

        # Lista med aktiva lager
        layers = [scatter_layer]
        if line_layer:
            layers.append(line_layer)

        # Kartvy
        view_state = pdk.ViewState(
            latitude=df["lat"].mean(),
            longitude=df["lon"].mean(),
            zoom=10.5,
            pitch=0,
        )

        # Visa kartan
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v11",
            initial_view_state=view_state,
            layers=layers,
            parameters={
                "dragPan": False,
                "scrollZoom": False,
                "doubleClickZoom": False,
                "touchZoom": False,
                "keyboard": False
            },
        ), use_container_width=False, width=300, height=450)
        
        st.markdown("---")
        st.markdown("""  
        VrångGo är skapad av studenter på Göteborgs Universitet som vill göra resandet i Göteborgs södra skärgård enklare och mer hållbart.
                    
            """)

   # st.map(data=None, *, latitude=None, longitude=None, color=None, size=None, zoom=None, use_container_width=True, width=None, height=None)

if __name__ == "__main__":
    main()