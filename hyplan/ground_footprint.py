from typing import List, Tuple
from shapely.geometry import Point
from pint import Quantity
import numpy as np
from hyplan.terrain import ray_terrain_intersection
from hyplan.units import ureg

def footprint_corners(
    lat: float, 
    lon: float, 
    altitude: float, 
    fov_x: float, 
    fov_y: float, 
    dem_file: str
) -> List[Tuple[Quantity, Quantity, Quantity]]:
    """
    Calculate the latitude, longitude, and altitude of the four corners of a FrameCamera's ground footprint.

    Args:
        lat (Quantity): Latitude of the FrameCamera in degrees.
        lon (Quantity): Longitude of the FrameCamera in degrees.
        altitude (Quantity): Altitude of the FrameCamera in meters.
        fov_x (Quantity): Horizontal Field of View (FoV) of the camera in degrees.
        fov_y (Quantity): Vertical Field of View (FoV) of the camera in degrees.
        dem_file (str): Path to the DEM file for terrain elevation data.

    Returns:
        List[Tuple[Quantity, Quantity, Quantity]]: A list of four tuples, each containing the latitude, 
            longitude, and altitude of a corner point of the ground footprint.
    """
    # Calculate the offsets in azimuth for the four corners
    azimuths = [45, 135, 225, 315]  # Diagonal directions for corners

    # Calculate the distances to the corners (half-width and half-height)
    half_width = altitude * np.tan(np.radians(fov_x / 2))
    half_height = altitude * np.tan(np.radians(fov_y / 2))

    # Combine these into corner distances
    corner_distances = [
        np.sqrt(half_width**2 + half_height**2)  # Same for all corners
    ] * 4

    # Calculate corner points
    corners = []
    for azimuth, distance in zip(azimuths, corner_distances):
        corner_lat, corner_lon, corner_alt = ray_terrain_intersection(
            lat=lat, lon=lon, altitude=altitude, azimuth=azimuth, distance=distance, dem_file=dem_file
        )
        corners.append((corner_lat, corner_lon, corner_alt))

    return corners

# Example usage
if __name__ == "__main__":
    lat = ureg.Quantity(34.0, "degree")
    lon = ureg.Quantity(-117.0, "degree")
    altitude = ureg.Quantity(5000, "meter")
    fov_x = ureg.Quantity(36.0, "degree")
    fov_y = ureg.Quantity(24.0, "degree")
    dem_file = "path/to/dem/file"

    corners = footprint_corners(lat, lon, altitude, fov_x, fov_y, dem_file)
    for idx, (corner_lat, corner_lon, corner_alt) in enumerate(corners):
        print(f"Corner {idx + 1}: Latitude={corner_lat:.6f}, Longitude={corner_lon:.6f}, Altitude={corner_alt:.2f} m")
