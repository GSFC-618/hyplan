#%%

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from hyplan.units import ureg
from hyplan.airports import Airport, initialize_data
from hyplan.dubins_path import Waypoint, DubinsPath
from hyplan.aircraft import Aircraft

initialize_data()  # Initialize airport data
# Define example airport and waypoints
example_airport = Airport(icao="KSBA")  # Fresno Yosemite International Airport
waypoint_1 = Waypoint(
    latitude=34.6928, longitude=-119.0406, heading=235.0, altitude=20000 * ureg.feet
)
waypoint_2 = Waypoint(
    latitude=34.5786, longitude=-119.5, heading=200.0, altitude=20000 * ureg.feet
)

# Define an example aircraft
example_aircraft = Aircraft(
    type="Beechcraft King Air 200",
    tail_number="N53W",
    service_ceiling=35_000 * ureg.feet,
    approach_speed=103 * ureg.knot,
    best_rate_of_climb=2_450 * ureg.feet / ureg.minute,
    cruise_speed=260 * ureg.knot,
    range=1_580 * ureg.nautical_mile,
    endurance=6 * ureg.hour,
    operator="Dynamic Aviation",
    useful_payload=4_250 * ureg.pound,
    descent_rate=1000 * ureg.feet / ureg.minute,
    vx=120 * ureg.knot,
    vy=135 * ureg.knot,
    max_bank_angle=25.0
)

# Flight phase calculations
takeoff_info = example_aircraft.time_to_takeoff(example_airport, waypoint_1)
cruise_info = example_aircraft.time_to_cruise(waypoint_1, waypoint_2)
return_info = example_aircraft.time_to_return(waypoint_2, example_airport)

# Extract coordinates using only DubinsPath
def extract_coordinates(path_data):
    if isinstance(path_data, DubinsPath):
        return [(lat, lon) for lon, lat in path_data.geometry.coords]  # Ensure correct order
    return []

takeoff_coords = extract_coordinates(takeoff_info["dubins_path"])
cruise_coords = extract_coordinates(cruise_info["dubins_path"])
return_coords = extract_coordinates(return_info["dubins_path"])

takeoff_lats, takeoff_lons = zip(*takeoff_coords) if takeoff_coords else ([], [])
cruise_lats, cruise_lons = zip(*cruise_coords) if cruise_coords else ([], [])
return_lats, return_lons = zip(*return_coords) if return_coords else ([], [])

# Convert to lists
takeoff_lons, takeoff_lats = list(takeoff_lons), list(takeoff_lats)
cruise_lons, cruise_lats = list(cruise_lons), list(cruise_lats)
return_lons, return_lats = list(return_lons), list(return_lats)

# Map extent calculation
all_lons = takeoff_lons + return_lons + cruise_lons
all_lats = takeoff_lats + return_lats + cruise_lats

if not all_lons or not all_lats or any(np.isnan(all_lons)) or any(np.isnan(all_lats)):
    raise ValueError("No valid coordinates for plotting.")

margin = 1  # Degrees of buffer around the flight path
lon_min, lon_max = min(all_lons) - margin, max(all_lons) + margin
lat_min, lat_max = min(all_lats) - margin, max(all_lats) + margin

#%%
# Plotting Ground Track
plt.figure(figsize=(12, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Airport and waypoint markers
ax.plot(example_airport.longitude, example_airport.latitude, marker='*', color='gold', markersize=15, transform=ccrs.PlateCarree(), label='Airport')
ax.plot(waypoint_1.longitude, waypoint_1.latitude, marker='^', color='purple', markersize=10, transform=ccrs.PlateCarree(), label='Waypoint 1')
ax.plot(waypoint_2.longitude, waypoint_2.latitude, marker='^', color='darkorange', markersize=10, transform=ccrs.PlateCarree(), label='Waypoint 2')

# Flight paths
ax.plot(takeoff_lons, takeoff_lats, label="Takeoff", color="blue", linewidth=2, transform=ccrs.PlateCarree())
ax.plot(cruise_lons, cruise_lats, label="Cruise", color="green", linewidth=2, transform=ccrs.PlateCarree())
ax.plot(return_lons, return_lats, label="Return", color="red", linewidth=2, transform=ccrs.PlateCarree())

plt.title("Ground Track Trajectory", fontsize=16)
plt.legend(fontsize=12)
plt.show()

#%%
# Plotting Altitude vs. Time
# Plotting Altitude vs. Time
# Plotting Altitude vs. Time with Sequential Time Adjustment
plt.figure(figsize=(10, 5))

phases = {**takeoff_info["phases"], **cruise_info["phases"], **return_info["phases"]}
colors = {"takeoff_climb": "blue", "takeoff_cruise": "cyan", "cruise": "green", "cruise_descent": "orange", "return_cruise": "purple", "return_descent": "red", "return_approach": "brown"}

# Adjust times for sequential alignment
adjusted_phases = {}
current_time = 0 * ureg.minute  # Initialize time

for phase in ["takeoff_climb", "takeoff_cruise", "cruise", "return_cruise", "return_descent", "return_approach"]:
    if phase in phases:
        start_altitude = phases[phase]["start_altitude"]
        end_altitude = phases[phase]["end_altitude"]
        duration = phases[phase]["end_time"] - phases[phase]["start_time"]

        adjusted_phases[phase] = {
            "start_time": current_time,
            "end_time": current_time + duration,
            "start_altitude": start_altitude,
            "end_altitude": end_altitude,
        }
        current_time += duration

# Plot adjusted phases
for phase, color in colors.items():
    if phase in adjusted_phases:
        plt.plot(
            [adjusted_phases[phase]["start_time"].magnitude, adjusted_phases[phase]["end_time"].magnitude],
            [adjusted_phases[phase]["start_altitude"].magnitude, adjusted_phases[phase]["end_altitude"].magnitude],
            color=color, marker="o", label=phase
        )

plt.xlabel("Time (minutes)")
plt.ylabel("Altitude (feet)")
plt.title("Altitude vs. Time Trajectory")
plt.legend()
plt.grid(True)
plt.show()

# %%
