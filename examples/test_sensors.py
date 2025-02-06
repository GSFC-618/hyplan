#%%
import matplotlib.pyplot as plt
import numpy as np
from hyplan.sensors import SENSOR_REGISTRY, create_sensor
from hyplan.units import ureg

altitudes = np.linspace(0, 10_000, 200) * ureg.m

sensor_names = SENSOR_REGISTRY.keys()
sensors = [create_sensor(name) for name in sensor_names]

swath_data = {sensor.name: [] for sensor in sensors}
critical_speed_data = {sensor.name: [] for sensor in sensors}
pixel_size_data = {sensor.name: [] for sensor in sensors}

for altitude in altitudes:
    for sensor in sensors:
        swath_data[sensor.name].append(sensor.swath_width(altitude).magnitude)
        critical_speed_data[sensor.name].append(sensor.critical_ground_speed(altitude).magnitude)
        pixel_size_data[sensor.name].append(sensor.ground_sample_distance(altitude).magnitude)

#%%

# Plot Swath Width vs Altitude
plt.figure(figsize=(10, 6))
for name, swaths in swath_data.items():
    plt.plot(altitudes.magnitude, swaths, label=name)
plt.title("Swath Width vs Altitude")
plt.xlabel("Altitude (m)")
plt.ylabel("Swath Width (m)")
plt.legend()
plt.grid()
plt.show()

# Plot Critical Ground Speed vs Altitude
plt.figure(figsize=(10, 6))
for name, speeds in critical_speed_data.items():
    plt.plot(altitudes.magnitude, speeds, label=name)
plt.title("Critical Ground Speed vs Altitude")
plt.xlabel("Altitude (m)")
plt.ylabel("Critical Ground Speed (m/s)")
plt.legend()
plt.grid()
plt.show()

# Plot Pixel Size vs Altitude
plt.figure(figsize=(10, 6))
for name, pixel_sizes in pixel_size_data.items():
    plt.plot(altitudes.magnitude, pixel_sizes, label=name)
plt.title("Ground Sample Distance vs Altitude")
plt.xlabel("Altitude (m)")
plt.ylabel("Pixel Size (m)")
plt.legend()
plt.grid()
plt.show()

# Plot altitude needed for a pixel size of 5m
desired_pixel_size = 5 * ureg.m

plt.figure(figsize=(10, 6))
for sensor in sensors:
    altitudes_for_pixel_size = sensor.altitude_for_ground_sample_distance(desired_pixel_size, desired_pixel_size).magnitude
    plt.bar(sensor.name, altitudes_for_pixel_size)

plt.title("Altitude Needed for Pixel Size of 5m")
plt.xlabel("Sensor")
plt.ylabel("Altitude (m)")
plt.xticks(rotation=45, ha="right")
plt.grid()
plt.show()

# %%
