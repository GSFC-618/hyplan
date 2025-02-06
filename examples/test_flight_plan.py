#%%
import geopandas as gpd
import matplotlib.pyplot as plt
from hyplan.aircraft import DynamicAviation_B200
from hyplan.airports import Airport
from hyplan.dubins_path import Waypoint
from hyplan.flight_line import FlightLine
from hyplan.flight_plan import compute_flight_plan, plot_flight_plan, plot_altitude_trajectory
from hyplan.units import ureg

# Define example aircraft
aircraft = DynamicAviation_B200()


# Define airports
departure_airport = Airport("KLAX")
return_airport = Airport("KSBA")

# Define waypoints
waypoint_1 = Waypoint(latitude=34.05, longitude=-118.25, heading=45.0, altitude=20000 * ureg.feet, name="WP1")
waypoint_2 = Waypoint(latitude=34.10, longitude=-118.20, heading=90.0, altitude=20000 * ureg.feet, name="WP2")
waypoint_3 = Waypoint(latitude=34.25, longitude=-118.00, heading=230.0, altitude=4000 * ureg.feet, name="WP2")

# Define flight lines
flight_line_1 = FlightLine.start_length_azimuth(
    lat1=34.10, lon1=-118.20,
    length=ureg.Quantity(50000, "meter"),
    az=0.0,
    altitude=ureg.Quantity(15000, "feet"),
    site_name="Test Flight Line 1"
)

flight_line_1 = flight_line_1.reverse()

flight_line_2 = FlightLine.start_length_azimuth(
    lat1=34.15, lon1=-118.10,
    length=ureg.Quantity(70000, "meter"),
    az=10.0,
    altitude=ureg.Quantity(18000, "feet"),
    site_name="Test Flight Line 2"
)

# Define flight sequence
flight_sequence = [flight_line_1, waypoint_3, flight_line_2, waypoint_1, waypoint_2]


#%%

# Compute flight plan
flight_plan_df = compute_flight_plan(aircraft, flight_sequence, departure_airport, return_airport)

# Display results
print(flight_plan_df)

#%%
# Plot flight plan
plot_flight_plan(flight_plan_df, departure_airport, return_airport, flight_sequence)
plot_altitude_trajectory(flight_plan_df)

# %%
