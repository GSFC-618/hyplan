#%%
import matplotlib.pyplot as plt
from hyplan.dubins_path import Waypoint, DubinsPath
from hyplan.units import ureg

# Define waypoints
waypoints = [
    Waypoint(latitude=34.05, longitude=-118.25, heading=0.0, altitude=100 * ureg.meter, name = "Waypoint 1"),
    Waypoint(latitude=34.10, longitude=-118.15, heading=45.0),
    Waypoint(latitude=34.15, longitude=-118.05, heading=135.0, name = "Waypoint 3"),
    Waypoint(latitude=34.20, longitude=-117.95, heading=0.0, name = "Waypoint 4"),
]

# Define parameters
speed = 50.0  # Speed in m/s
bank_angle = 5.0  # Bank angle in degrees
step_size = 10.0  # Step size in meters

# Generate Dubins paths between consecutive waypoints
paths = []
for i in range(len(waypoints) - 1):
    dubins_path = DubinsPath(
        start=waypoints[i],
        end=waypoints[i + 1],
        speed=speed,
        bank_angle=bank_angle,
        step_size=step_size
    )
    paths.append(dubins_path)

# Plot the Dubins paths
plt.figure(figsize=(10, 6))
for i, path in enumerate(paths):
    lons, lats = zip(*list(path.geometry.coords))
    plt.plot(lons, lats, label=f"Path {i + 1}")

# Plot waypoints
for waypoint in waypoints:
    plt.scatter(waypoint.geometry.x, waypoint.geometry.y, color='red', zorder=5)
    plt.text(waypoint.geometry.x, waypoint.geometry.y, waypoint.name, fontsize=9)

# Add plot details
plt.title("Dubins Paths Between Waypoints")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid()
plt.show()

# %%
