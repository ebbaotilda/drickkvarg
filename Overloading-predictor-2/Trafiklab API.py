import flet as ft
import requests
import json
from datetime import datetime, timedelta

# ResRobot API-nyckel
access_id = "25add2dc-d84e-4119-89f3-091d4c78c3d5"  # Din API-nyckel
saltholmen_id = "740001206"  # Saltholmen
vrango_id = "740001382"  # Vrångö

# Hämta reseplaner från ResRobot
def get_trips(origin_id, dest_id, date, time=None):
    url = "https://api.resrobot.se/v2.1/trip"
    results = set()  # För att undvika dubletter
    
    if time:
        # Om tid är angiven, hämta närmaste avgångar efter den tiden
        params = {
            "format": "json",
            "originId": origin_id,
            "destId": dest_id,
            "date": date,
            "time": time,
            "numF": 3,
            "passlist": 1,
            "accessId": access_id
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            print(f"Rådata för tid {time}: {response.text[:500]}...")
            data = response.json()
            trips = data.get("Trip", [])
            if not trips:
                print(f"Inga resor hittades för tid {time}")
            for trip in trips:
                for leg in trip.get("LegList", {}).get("Leg", []):
                    if leg["type"] == "JNY":
                        product = leg.get("Product", [{}])[0]
                        if "281" in product.get("num", ""):
                            departure = f"{leg['Origin']['date']} {leg['Origin']['time']}"
                            results.add((
                                departure,
                                product.get("num", "N/A"),
                                leg.get("direction", "N/A"),
                                f"{leg['Origin']['date']} {leg['Origin']['time']}",
                                f"{leg['Destination']['date']} {leg['Destination']['time']}"
                            ))
        except Exception as e:
            print(f"Fel vid hämtning för tid {time}: {str(e)}")
            return [{"error": str(e)}]
    else:
        # Om ingen tid är angiven, hämta alla resor för dagen med flera anrop
        times = ["00:00", "06:00", "12:00", "18:00"]
        for time in times:
            params = {
                "format": "json",
                "originId": origin_id,
                "destId": dest_id,
                "date": date,
                "time": time,
                "numF": 6,  # Sänkt till 6 för att undvika 400 Bad Request
                "passlist": 1,
                "accessId": access_id
            }
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                print(f"Rådata för tid {time}: {response.text[:500]}...")
                data = response.json()
                trips = data.get("Trip", [])
                if not trips:
                    print(f"Inga resor hittades för tid {time}")
                    continue
                for trip in trips:
                    for leg in trip.get("LegList", {}).get("Leg", []):
                        if leg["type"] == "JNY":
                            product = leg.get("Product", [{}])[0]
                            if "281" in product.get("num", ""):
                                departure = f"{leg['Origin']['date']} {leg['Origin']['time']}"
                                print(f"Hittade resa: Linje {product.get('num')}, Avgång {departure}")
                                results.add((
                                    departure,
                                    product.get("num", "N/A"),
                                    leg.get("direction", "N/A"),
                                    f"{leg['Origin']['date']} {leg['Origin']['time']}",
                                    f"{leg['Destination']['date']} {leg['Destination']['time']}"
                                ))
            except Exception as e:
                print(f"Fel vid hämtning för tid {time}: {str(e)}")
                return [{"error": str(e)}]
    
    print(f"Totalt antal unika resor hittade: {len(results)}")
    # Konvertera till lista och sortera efter avgångstid
    results = [
        {
            "line": line,
            "direction": direction,
            "departure": departure,
            "arrival": arrival
        }
        for departure, line, direction, departure, arrival in sorted(results)
    ]
    return results if results else [{"error": "Inga resor hittades för den valda dagen. Kontrollera loggarna för mer information."}]

# Flet-applikation
def main(page: ft.Page):
    page.title = "Tidtabell för linje 281"
    page.padding = 20
    page.window_width = 600
    page.window_height = 800

    # Dropdown för riktning
    direction = ft.Dropdown(
        label="Riktning",
        options=[
            ft.dropdown.Option(key="vrango_to_saltholmen", text="Vrångö till Saltholmen"),
            ft.dropdown.Option(key="saltholmen_to_vrango", text="Saltholmen till Vrångö")
        ],
        value="vrango_to_saltholmen",
        width=300
    )

    # Dropdown för datum (7 dagar framåt)
    dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    date = ft.Dropdown(
        label="Datum",
        options=[ft.dropdown.Option(key=d, text=d) for d in dates],
        value=dates[0],
        width=300
    )

    # Dropdown för tid (valfritt)
    hours = [f"{h:02d}:00" for h in range(24)]
    time = ft.Dropdown(
        label="Tid (valfritt, välj för närmaste avgångar)",
        options=[ft.dropdown.Option(key=None, text="Hela dagen")] + [ft.dropdown.Option(key=h, text=h) for h in hours],
        value=None,
        width=300
    )

    # Resultatlista
    results_list = ft.ListView(expand=True, spacing=10, padding=10)

    # Laddningsindikator
    loading_text = ft.Text("Hämtar tidtabeller...", visible=False, color=ft.Colors.BLUE)

    # Knapp för att hämta tidtabeller
    def fetch_timetables(e):
        results_list.controls.clear()
        loading_text.visible = True
        page.update()
        origin_id = vrango_id if direction.value == "vrango_to_saltholmen" else saltholmen_id
        dest_id = saltholmen_id if direction.value == "vrango_to_saltholmen" else vrango_id
        selected_time = time.value if time.value != "None" else None
        trips = get_trips(origin_id, dest_id, date.value, selected_time)
        loading_text.visible = False
        if trips and "error" in trips[0]:
            results_list.controls.append(ft.Text(f"Fel: {trips[0]['error']}", color=ft.colors.RED))
        else:
            for trip in trips:
                results_list.controls.append(
                    ft.Card(
                        content=ft.Container(
                            content=ft.Column([
                                ft.Text(f"Linje: {trip['line']}", weight=ft.FontWeight.BOLD),
                                ft.Text(f"Riktning: {trip['direction']}"),
                                ft.Text(f"Avgång: {trip['departure']}"),
                                ft.Text(f"Ankomst: {trip['arrival']}")
                            ]),
                            padding=10
                        )
                    )
                )
        page.update()

    fetch_button = ft.ElevatedButton("Hämta tidtabeller", on_click=fetch_timetables)

    # Lägg till komponenter till sidan
    page.add(
        ft.Column([
            ft.Text("Välj riktning, datum och tid för linje 281", size=20, weight=ft.FontWeight.BOLD),
            direction,
            date,
            time,
            fetch_button,
            loading_text,
            results_list
        ])
    )

if __name__ == "__main__":
    ft.app(target=main)