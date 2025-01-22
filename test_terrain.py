# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import logging
from hyplan.terrain import (
    get_cache_root,
    clear_cache,
    clear_localdem_cache,
    generate_demfile,
    get_elevations,
    get_min_max_elevations,
    ray_terrain_intersection,
)
from hyplan.geometry import wrap_to_180
import pymap3d.vincenty

gdal.UseExceptions()

# Set up logging
logging.basicConfig(level=logging.INFO)

# %% Test cache clearing functions
print("Testing cache clearing functions...")

# Clear the entire cache
try:
    clear_cache()
    print("Cache cleared successfully.")
except ValueError as e:
    print(f"Error clearing cache: {e}")

# Clear only the local DEM cache
try:
    clear_localdem_cache(confirm=True)  # Set confirm=False for automated testing
    print("Local DEM cache cleared successfully.")
except Exception as e:
    print(f"Error clearing local DEM cache: {e}")

# %% Define the bounding box for the region of interest
lat_min, lat_max = 33.9, 34.2  # Example latitudes (Los Angeles area)
lon_min, lon_max = -118.4, -118.1  # Example longitudes (Los Angeles area)
latitude = np.array([lat_min, lat_max])
longitude = np.array([lon_min, lon_max])

# %% Generate the DEM file
print("Generating DEM file for the specified region...")
try:
    dem_file = generate_demfile(latitude, longitude)
    print(f"DEM file generated: {dem_file}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

print("Cache directory: ", get_cache_root())

# %% Generate random points within the bounding box
print("Generating random points within the bounding box...")
num_points = 100
random_lats = np.random.uniform(lat_min, lat_max, num_points)
random_lons = np.random.uniform(lon_min, lon_max, num_points)

# %% Extract elevation for random points
print("Extracting elevation values for random points...")
try:
    elevations = get_elevations(random_lats, random_lons, dem_file)
    for i in range(min(10, num_points)):  # Display the first 10 points
        print(f"Point {i + 1}: Lat={random_lats[i]:.5f}, Lon={random_lons[i]:.5f}, Elevation={elevations[i]:.2f} meters")
    if num_points > 10:
        print(f"...and {num_points - 10} more points.")
except Exception as e:
    print(f"Error extracting elevation: {e}")
    exit(1)

# %% Visualize the DEM and elevation points
print("Visualizing the DEM and elevation points in 2D...")
try:
    dem_dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
    dem_band = dem_dataset.GetRasterBand(1)
    dem_data = dem_band.ReadAsArray()
    dem_transform = dem_dataset.GetGeoTransform()

    # Downsample the DEM data for faster plotting
    step = 10
    dem_data_downsampled = dem_data[::step, ::step]
    cols, rows = dem_data_downsampled.shape

    # Create a grid of coordinates for the downsampled DEM
    x = np.arange(0, cols * step, step) * dem_transform[1] + dem_transform[0]
    y = np.arange(0, rows * step, step) * dem_transform[5] + dem_transform[3]
    x, y = np.meshgrid(x, y)

    # Plot the DEM
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.imshow(
        dem_data_downsampled,
        cmap="terrain",
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="upper",
    )
    plt.colorbar(c, ax=ax, label="Elevation (meters)")

    # Plot the random elevation points
    ax.scatter(random_lons, random_lats, c="red", label="Random Points", edgecolor="black")
    ax.set_title("DEM with Elevation Points (Optimized 2D)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.grid()
    plt.show()
except Exception as e:
    print(f"Error visualizing the DEM in 2D: {e}")

# %% Display min and max elevations from the DEM
print("Min and Max elevations from the DEM:")
try:
    min_elev, max_elev = get_min_max_elevations(dem_file)
    print(f"Min Elevation: {min_elev} meters, Max Elevation: {max_elev} meters")
except Exception as e:
    print(f"Error fetching min/max elevations: {e}")

# %% Test ray-terrain intersection
print("Testing ray-terrain intersection function...")
lat_start, lon_start = 34.25, -117.75  # Starting point (latitude, longitude)
lat_end, lon_end = 34.50, -117.75  # Ending point (latitude, longitude)
num_observers = 300
h0 = 5000  # Altitude of observer (meters)
az = 90  # Azimuth angle (degrees)
tilt = 17  # Tilt angle (degrees)
precision = 10.0  # Precision of slant range sampling (meters)

# Generate observer positions along a straight line
lats, lons = pymap3d.vincenty.track2(lat_start, lon_start, lat_end, lon_end, npts=num_observers)
lons = wrap_to_180(lons)

# Generate DEM for the region
latitude = np.array([min(lats), max(lats)])
longitude = np.array([min(lons), max(lons)])
dem_file = generate_demfile(latitude, longitude)

try:
    intersection_lats, intersection_lons, intersection_alts = ray_terrain_intersection(
        lats,
        lons,
        h0,
        az,
        tilt,
        precision,
        dem_file,
    )

    # Display the results
    for i in range(min(10, len(lats))):  # Show first 10 results
        print(f"Observer {i + 1}:")
        print(f"  Latitude: {lats[i]:.6f}")
        print(f"  Longitude: {lons[i]:.6f}")
        print(f"  Intersection Latitude: {intersection_lats[i]:.6f}")
        print(f"  Intersection Longitude: {intersection_lons[i]:.6f}")
        print(f"  Intersection Elevation: {intersection_alts[i]:.2f} meters")

    # Visualize the intersections
    plt.figure(figsize=(10, 8))
    plt.plot(lons, lats, '-', label="Observers")
    plt.plot(intersection_lons, intersection_lats, c="red", label="Intersections")
    plt.title("Ray-Terrain Intersections")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid()
    plt.show()
except Exception as e:
    print(f"Error testing ray-terrain intersection: {e}")

# %%
