#%%

import numpy as np
import matplotlib.pyplot as plt
from hyplan.units import ureg
from hyplan.airports import Airport, initialize_data
from hyplan.dubins_path import Waypoint
from hyplan.aircraft import Aircraft

initialize_data()  # Initialize airport data
# Define example airport and waypoint
example_airport = Airport(icao="KLAX")  # Los Angeles International Airport  # Los Angeles International Airport
example_waypoint = Waypoint(
    latitude=34.5, longitude=-118.5, heading=90, altitude=10_000 * ureg.feet
)

# Define an example aircraft
example_aircraft = Aircraft(
    type="Beechcraft King Air 200",
    tail_number="N53W",
    service_ceiling=35_000 * ureg.feet,
    approach_speed=103 * ureg.knot,
    best_rate_of_climb=2_450 * ureg.feet / ureg.minute,
    cruise_speed=270 * ureg.knot,
    range=1_580 * ureg.nautical_mile,
    endurance=6 * ureg.hour,
    operator="Dynamic Aviation",
    useful_payload=4_250 * ureg.pound,
    descent_rate=1_500 * ureg.feet / ureg.minute,
    vx=120 * ureg.knot,
    vy=135 * ureg.knot,
)

# Get detailed flight phase information for takeoff
takeoff_info = example_aircraft.time_to_takeoff(example_airport, example_waypoint)
takeoff_phases = takeoff_info["phases"]
takeoff_path = takeoff_info["dubins_path"]

# Get detailed flight phase information for return
return_info = example_aircraft.time_to_return(example_waypoint, example_airport)
return_phases = return_info["phases"]
return_path = return_info["dubins_path"]

#%%
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np

# Adjust return phase times to be sequential
time_offset = float(takeoff_phases["takeoff_cruise"]["end_time"].to("minute").magnitude)
adjusted_return_phases = {}
for phase, values in return_phases.items():
    adjusted_return_phases[phase] = {
        "start_time": float(values["start_time"].to("minute").magnitude + time_offset),
        "end_time": float(values["end_time"].to("minute").magnitude + time_offset),
        "start_altitude": float(values["start_altitude"].to("foot").magnitude),
        "end_altitude": float(values["end_altitude"].to("foot").magnitude),
    }

# Define colors for each phase
phase_colors = {
    "takeoff_climb": "blue",
    "takeoff_cruise": "cyan",
    "return_cruise": "green",
    "return_descent": "orange",
    "return_approach": "red",
}

# Extract altitude and time data
time_values = []
altitude_values = []
labels = []

for phase in ["takeoff_climb", "takeoff_cruise", "return_cruise", "return_descent", "return_approach"]:
    phase_data = takeoff_phases.get(phase, adjusted_return_phases.get(phase))
    time_values.extend([float(phase_data["start_time"]), float(phase_data["end_time"])])
    altitude_values.extend([float(phase_data["start_altitude"]), float(phase_data["end_altitude"])])
    labels.append(phase)

# Convert time and altitude lists to numpy arrays
time_values = np.array(time_values, dtype=float)
altitude_values = np.array(altitude_values, dtype=float)

# Plot altitude vs. time
plt.figure(figsize=(10, 5))
prev_end = 0
for phase, color in phase_colors.items():
    phase_data = takeoff_phases.get(phase, adjusted_return_phases.get(phase))
    plt.plot(
        [float(phase_data["start_time"]), float(phase_data["end_time"])],
        [float(phase_data["start_altitude"]), float(phase_data["end_altitude"])],
        color=color, marker="o", label=phase
    )
plt.xlabel("Time (minutes)")
plt.ylabel("Altitude (feet)")
plt.title("Altitude vs. Time Trajectory")
plt.legend()
plt.grid(True)
plt.show()

# Extract Dubins path ground track

def extract_coordinates(dubins_path):
    if isinstance(dubins_path.geometry, LineString):
        return list(dubins_path.geometry.coords)
    return []

takeoff_coords = extract_coordinates(takeoff_info["dubins_path"])
return_coords = extract_coordinates(return_info["dubins_path"])

takeoff_lats, takeoff_lons = zip(*[(lat, lon) for lon, lat in takeoff_coords])
return_lats, return_lons = zip(*[(lat, lon) for lon, lat in return_coords])

# Plot ground track
plt.figure(figsize=(10, 5))
plt.plot(takeoff_lons, takeoff_lats, label="Takeoff Ground Track", color="blue", marker="o")
plt.plot(return_lons, return_lats, label="Return Ground Track", color="red", marker="o")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Ground Track Trajectory")
plt.legend()
plt.grid(True)
plt.show()
