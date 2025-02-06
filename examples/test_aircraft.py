#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from hyplan.units import ureg
from hyplan.airports import Airport, initialize_data
from hyplan.dubins_path import Waypoint, DubinsPath
from hyplan.aircraft import Aircraft
from hyplan.flight_line import FlightLine

# Initialize airport data
initialize_data()

# Define example airport and flight line
airport = Airport(icao="KSBA")  # Example airport: Santa Barbara Municipal
flight_line = FlightLine.start_length_azimuth(
    lat1=34.05,
    lon1=-118.25,
    length=ureg.Quantity(100000, "meter"),
    az=45.0,
    altitude=ureg.Quantity(20000, "feet"),
    site_name="LA Northeast",
    investigator="Dr. Smith"
)

waypoint_1 = flight_line.waypoint1
waypoint_2 = flight_line.waypoint2

# Define an example aircraft
aircraft = Aircraft(
    type="Beechcraft King Air 200",
    tail_number="N53W",
    service_ceiling=35000 * ureg.feet,
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
    max_bank_angle=30.0
)

# Compute flight phases
takeoff_info = aircraft.time_to_takeoff(airport, waypoint_1)
cruise_info = aircraft.time_to_cruise(waypoint_1, waypoint_2)
return_info = aircraft.time_to_return(waypoint_2, airport)

# Helper function to extract coordinates from DubinsPath
def extract_coordinates(dubins_path):
    coords = [(lat, lon) for lon, lat in dubins_path.geometry.coords]
    return zip(*coords) if coords else ([], [])

# Plot Ground Track
def plot_ground_track(phases, title="Ground Track"):
    plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot phases
    colors = {"Takeoff": "blue", "Cruise": "green", "Return": "red"}
    for name, phase_info in phases.items():
        lats, lons = extract_coordinates(phase_info["dubins_path"])
        ax.plot(lons, lats, label=name, color=colors[name], linewidth=2, transform=ccrs.PlateCarree())

    # Plot airport and waypoints
    ax.plot(airport.longitude, airport.latitude, marker="*", color="gold", markersize=15, label="Airport", transform=ccrs.PlateCarree())
    ax.plot(waypoint_1.longitude, waypoint_1.latitude, marker="^", color="purple", markersize=10, label="Waypoint 1", transform=ccrs.PlateCarree())
    ax.plot(waypoint_2.longitude, waypoint_2.latitude, marker="^", color="orange", markersize=10, label="Waypoint 2", transform=ccrs.PlateCarree())

    plt.title(title)
    plt.legend()
    plt.show()

# Plot Altitude Trajectory
def plot_altitude_trajectory(phases, title="Altitude Trajectory"):
    plt.figure(figsize=(10, 5))
    current_time = 0 * ureg.minute

    for name, phase_info in phases.items():
        for sub_phase_name, phase_data in phase_info["phases"].items():
            start_alt = phase_data["start_altitude"].to("feet").magnitude
            end_alt = phase_data["end_altitude"].to("feet").magnitude
            duration = (phase_data["end_time"] - phase_data["start_time"]).to("minute").magnitude

            # Ensure duration is treated as a pint.Quantity
            duration = (phase_data["end_time"] - phase_data["start_time"]).to("minute")

            # Update the plot line to handle pint.Quantity
            plt.plot(
                [current_time.to("minute").magnitude, (current_time + duration).to("minute").magnitude],
                [start_alt, end_alt],
                label=f"{name} - {sub_phase_name}"
            )

            # Increment current_time correctly
            current_time += duration


    plt.xlabel("Time (minutes)")
    plt.ylabel("Altitude (feet)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


# Incremental plotting
# 1. Takeoff only
plot_ground_track({"Takeoff": takeoff_info}, "Ground Track: Takeoff Only")
plot_altitude_trajectory({"Takeoff": takeoff_info}, "Altitude Trajectory: Takeoff Only")

# 2. Takeoff and Cruise
plot_ground_track({"Takeoff": takeoff_info, "Cruise": cruise_info}, "Ground Track: Takeoff and Cruise")
# plot_altitude_trajectory({"Takeoff": takeoff_info, "Cruise": cruise_info}, "Altitude Trajectory: Takeoff and Cruise")

# 3. Full Flight (Takeoff, Cruise, and Return)
plot_ground_track({"Takeoff": takeoff_info, "Cruise": cruise_info, "Return": return_info}, "Ground Track: Full Flight")
plot_altitude_trajectory({"Takeoff": takeoff_info, "Cruise": cruise_info, "Return": return_info}, "Altitude Trajectory: Full Flight")

# %%
