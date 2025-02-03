from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pymap3d import los
from sunposition import sunpos

from .flight_line import FlightLine
from .sensors import LineScanner
from .geometry import process_linestring

def calculate_target_and_glint_vectorized(
    sensor_lat, sensor_lon, sensor_alt,
    viewing_azimuth, tilt_angle,
    observation_datetime
):
    """
    Vectorized calculation of target locations and glint angles for a set of sensors.

    Args:
        sensor_lat (np.ndarray): Latitudes of the sensors (decimal degrees).
        sensor_lon (np.ndarray): Longitudes of the sensors (decimal degrees).
        sensor_alt (np.ndarray): Altitudes of the sensors above sea level (meters).
        viewing_azimuth (np.ndarray): Viewing azimuths relative to true north (degrees).
        tilt_angle (np.ndarray): Tilt angles of the sensors from nadir (degrees).
        observation_datetime (np.ndarray): Timestamps of observations (UTC).

    Returns:
        tuple: (target_lat, target_lon, glint_angles)
    """
    # Ensure observation_datetime is a NumPy array
    observation_datetime = np.asarray(observation_datetime)

    # Step 1: Calculate the target locations on the surface using pymap3d
    target_lat, target_lon, _ = los.lookAtSpheroid(
        sensor_lat,
        sensor_lon,
        sensor_alt,
        viewing_azimuth,
        tilt_angle
    )

    # Step 2: Calculate the solar positions using `sunpos`
    solar_azimuth, solar_zenith, _, _, _ = sunpos(
        dt=observation_datetime,
        latitude=sensor_lat,
        longitude=sensor_lon,
        elevation=sensor_alt,
        radians=False  # Output in degrees
    )

    # Step 3: Calculate the glint angles
    glint_angles = glint_angle(solar_azimuth, solar_zenith, viewing_azimuth, np.abs(tilt_angle))

    return target_lat, target_lon, glint_angles

def glint_angle(solar_azimuth, solar_zenith, view_azimuth, view_zenith):
    """
    Calculates glint angles for each pixel based on solar and sensor angles.
    """
    solar_zenith_rad = np.deg2rad(solar_zenith)
    solar_azimuth_rad = np.deg2rad(solar_azimuth)
    view_zenith_rad = np.deg2rad(view_zenith)
    view_azimuth_rad = np.deg2rad(view_azimuth)

    phi = solar_azimuth_rad - view_azimuth_rad
    glint_cos = (
        np.cos(view_zenith_rad) * np.cos(solar_zenith_rad) -
        np.sin(view_zenith_rad) * np.sin(solar_zenith_rad) * np.cos(phi)
    )

    # Clamp to [-1, 1] to avoid numerical issues
    glint_cos = np.clip(glint_cos, -1, 1)

    glint_array = np.degrees(np.arccos(glint_cos))
    return glint_array

def compute_glint_vectorized(flight_line: FlightLine, sensor: LineScanner, observation_datetime, output_geometry="geographic"):
    """
    Computes glint angles across a flight line and returns the results as a GeoDataFrame.

    Args:
        flight_line (FlightLine): FlightLine object defining the flight path.
        sensor (LineScanner): LineScanner object defining sensor characteristics.
        observation_datetime (datetime): The observation timestamp.

    Returns:
        GeoDataFrame: Results containing target locations and glint angles.
    """
    # Get track coordinates, altitude, and azimuth
    latitudes, longitudes, azimuths, along_track_distance = process_linestring(flight_line.track())  # Get latitudes, longitudes, azimuths
    altitude = flight_line.altitude.magnitude  # Extract altitude magnitude

    # Define tilt angles from -half_angle to +half_angle in 1-degree increments
    half_angle = sensor.half_angle.magnitude  # Extract half angle from sensor
    tilt_angles = np.arange(-half_angle, half_angle + 1, 1)  # Shape: (T,)

    # Repeat azimuths for all tilt angles
    view_azimuths = np.repeat(azimuths+90.0, len(tilt_angles))

    # Tile tilt angles for all azimuths
    tilt_angles = np.tile(tilt_angles, len(latitudes))  # One azimuth per lat/lon

    # Repeat latitudes, longitudes, and altitudes to match the number of angle combinations
    latitudes = np.repeat(latitudes, len(tilt_angles) // len(latitudes))
    longitudes = np.repeat(longitudes, len(tilt_angles) // len(longitudes))
    altitudes = np.full_like(latitudes, altitude)
    along_track_distance = np.repeat(along_track_distance, len(tilt_angles) // len(along_track_distance))
    
    # Expand observation_datetime to match the shape of latitudes
    observation_datetimes = np.full(latitudes.shape, observation_datetime)

    # Call the vectorized glint calculation function
    target_lat, target_lon, glint_angles = calculate_target_and_glint_vectorized(
        sensor_lat=latitudes,
        sensor_lon=longitudes,
        sensor_alt=altitudes,
        viewing_azimuth=view_azimuths,
        tilt_angle=tilt_angles,
        observation_datetime=observation_datetimes
    )

    # Include tilt_angle and viewing_azimuth in the GeoDataFrame
    data = {
        "target_latitude": target_lat,
        "target_longitude": target_lon,
        "glint_angle": glint_angles,
        "tilt_angle": tilt_angles,
        "viewing_azimuth": view_azimuths,
        "along_track_distance": along_track_distance
    }

    if output_geometry == "geographic":
        geometry = [Point(lon, lat) for lon, lat in zip(target_lon, target_lat)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")  # Assuming WGS84 CRS
    elif output_geometry == "along_track":
        geometry = [Point(tilt_angles, along_track_distance) for tilt_angles,along_track_distance  in zip(tilt_angles,along_track_distance)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=None)  # Assuming WGS84 CRS
    else:
        raise ValueError("Invalid output_geometry parameter. Must be 'geographic' or 'along_track'.")

    return gdf
