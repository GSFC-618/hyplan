#%%
import matplotlib.pyplot as plt
import numpy as np
from hyplan.sensors import (
    AVIRISClassic, AVIRISNextGen, AVIRIS3, HyTES, PRISM, MASTER,
    GLiHT_VNIR, GLiHT_Thermal, GLiHT_SIF, CFIS
)
from hyplan.units import ureg

# Create instances of all spectrometers
spectrometers = [
    AVIRISClassic(), AVIRISNextGen(), AVIRIS3(), HyTES(), PRISM(),
    MASTER(), GLiHT_VNIR(), GLiHT_Thermal(), GLiHT_SIF(), CFIS()
]

# Define altitude range (0 to 10 km)
altitudes = np.linspace(0, 10_000, 200) * ureg.m

# Prepare dictionaries for data storage
swath_data = {spectrometer.name: [] for spectrometer in spectrometers}
critical_speed_data = {spectrometer.name: [] for spectrometer in spectrometers}
pixel_size_data = {spectrometer.name: [] for spectrometer in spectrometers}

# Populate data for each spectrometer
for altitude in altitudes:
    for spectrometer in spectrometers:
        swath_data[spectrometer.name].append(spectrometer.swath_width_at(altitude).magnitude)
        critical_speed_data[spectrometer.name].append(
            spectrometer.critical_ground_speed(altitude).magnitude
        )
        pixel_size_data[spectrometer.name].append(spectrometer.pixel_size_at(altitude).magnitude)

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
plt.title("Pixel Size vs Altitude")
plt.xlabel("Altitude (m)")
plt.ylabel("Pixel Size (m)")
plt.legend()
plt.grid()
plt.show()

# Plot altitude needed for a pixel size of 5m
desired_pixel_size = 5 * ureg.m

plt.figure(figsize=(10, 6))
for spectrometer in spectrometers:
    altitudes_for_pixel_size = spectrometer.alt_for_pixel_size(desired_pixel_size).magnitude
    plt.bar(spectrometer.name, altitudes_for_pixel_size)

plt.title("Altitude Needed for Pixel Size of 5m")
plt.xlabel("Spectrometer")
plt.ylabel("Altitude (m)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# %%
