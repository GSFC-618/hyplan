import numpy as np
import math
import random
import logging
from typing import Optional, Tuple, Callable

from shapely.affinity import affine_transform, translate
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import triangulate, transform, unary_union
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj import Transformer


# Wrap angles to -180 to 180 degrees
def wrap_to_180(lon):
    lon = np.mod(np.array(lon) + 180.0, 360.0) - 180.0
    return np.squeeze(lon)

def _validate_polygon(polygon: Optional[Polygon]) -> None:
    """
    Validate the input polygon and ensure it is a single, non-empty, valid Shapely Polygon.

    Args:
        polygon (Optional[Polygon]): The polygon to validate. Can be None.

    Raises:
        ValueError: If the polygon is invalid for any of the following reasons:
            - It is not a Shapely Polygon.
            - It is a MultiPolygon.
            - It is empty or has insufficient points.
            - It is invalid (e.g., self-intersecting).

    Returns:
        None: This function performs validation only and raises an error if the input is invalid.

    Notes:
        - If the polygon is None, the function assumes no validation is needed.
        - Uses Shapely's built-in validation for geometry validity checks.
    """
    if polygon is None:
        logging.debug("Polygon validation skipped because input is None.")
        return  # No validation needed for None

    if not isinstance(polygon, Polygon):
        if isinstance(polygon, MultiPolygon):
            raise ValueError(
                "MultiPolygon input is not supported. Provide a single Polygon."
            )
        raise ValueError(f"Input must be a Shapely Polygon. Received type: {type(polygon)}.")

    if polygon.is_empty:
        raise ValueError("Input polygon is empty.")

    if len(polygon.exterior.coords) < 4:
        raise ValueError(
            "Input polygon has insufficient points to form a valid geometry."
        )

    if not polygon.is_valid:
        raise ValueError(
            f"Input polygon is invalid: {polygon.explain_validity()}"
        )

    logging.debug("Polygon validation passed.")
    return True

def get_utm_crs(lon: float, lat: float) -> CRS:
    """
    Determine the UTM CRS for a given WGS84 coordinate using the area of interest (AOI).

    Args:
        lon (float): Longitude in decimal degrees (WGS84).
        lat (float): Latitude in decimal degrees (WGS84).

    Returns:
        CRS: The appropriate UTM CRS for the coordinate.
    """
    # Create an area of interest centered on the input coordinates
    aoi = AreaOfInterest(west_lon_degree=lon, south_lat_degree=lat,
                         east_lon_degree=lon, north_lat_degree=lat)

    # Query UTM CRS info for the area of interest
    utm_crs_list = query_utm_crs_info(datum_name="WGS 84", area_of_interest=aoi)

    # Return the first matching UTM CRS
    if not utm_crs_list:
        raise ValueError(f"No UTM CRS found for the coordinate ({lon}, {lat}).")
    
    return CRS.from_epsg(utm_crs_list[0].code)

def get_utm_transforms(geometry) -> Tuple[Callable, Callable]:
    """
    Get the UTM CRS and transformation functions to/from WGS84 for a Shapely geometry.

    Args:
        geometry (shapely.geometry.base.BaseGeometry): A Shapely geometry object. Must have a valid centroid.

    Returns:
        Tuple[Callable, Callable]: Transformation functions:
            - `wgs84_to_utm`: Function to transform coordinates from WGS84 to UTM.
            - `utm_to_wgs84`: Function to transform coordinates from UTM to WGS84.

    Raises:
        ValueError: If the geometry is invalid or has no valid centroid.
    """
    if not geometry.is_valid:
        raise ValueError("Invalid geometry provided.")
    if geometry.is_empty:
        raise ValueError("Empty geometry provided. Cannot calculate centroid.")

    # Extract centroid coordinates
    centroid = geometry.centroid
    if centroid.is_empty:
        raise ValueError("Geometry has no valid centroid.")

    lon, lat = centroid.x, centroid.y

    try:
        # Get UTM CRS based on the centroid
        utm_crs = get_utm_crs(lon, lat)
    except ValueError as e:
        raise ValueError(f"Failed to determine UTM CRS for centroid ({lon}, {lat}): {e}")

    # Define transformation functions
    wgs84_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform
    utm_to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform

    logging.debug(f"Generated UTM transformations for centroid ({lat:.6f}, {lon:.6f}).")
    return wgs84_to_utm, utm_to_wgs84

def haversine(lat1: float, lon1: float, lat2: float, lon2: float, radius: float = 6371e3) -> float:
    """
    Calculate the haversine distance between two points on the Earth's surface.

    Args:
        lat1 (float): Latitude of the first point in decimal degrees.
        lon1 (float): Longitude of the first point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.
        radius (float): Radius of the Earth in meters (default: 6371e3 for meters).

    Returns:
        float: Distance between the two points in the same unit as the radius.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c

def random_points_in_polygon(polygon, k):
    "Return list of k points chosen uniformly at random inside the polygon."
    areas = []
    transforms = []
    for t in triangulate(polygon):
        areas.append(t.area)
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        transforms.append([x1 - x0, x2 - x0, y2 - y0, y1 - y0, x0, y0])
    points = []
    for transform in random.choices(transforms, weights=areas, k=k):
        x, y = [random.random() for _ in range(2)]
        if x + y > 1:
            p = Point(1 - x, 1 - y)
        else:
            p = Point(x, y)
        points.append(affine_transform(p, transform))
    return points




def minimum_rotated_rectangle(polygon: Polygon) -> tuple:
    """
    Calculate the minimum rotated rectangle of a polygon in WGS84 coordinates.

    Args:
        polygon (Polygon): Input polygon in WGS84 coordinates. Must be valid.

    Returns:
        tuple: A tuple containing:
            - lat0 (float): Latitude of the rectangle's centroid.
            - lon0 (float): Longitude of the rectangle's centroid.
            - azimuth (float): Azimuth of the rectangle in degrees, wrapped to [-180, 180].
            - length (float): Length of the rectangle's longer side (meters).
            - width (float): Width of the rectangle's shorter side (meters).
            - mrr_wgs84 (Polygon): Minimum rotated rectangle in WGS84 coordinates.
            - hull_wgs84 (Polygon): Convex hull of the polygon in WGS84 coordinates.

    Raises:
        ValueError: If the input polygon is invalid or processing fails.

    Notes:
        - The input polygon is transformed to UTM for accurate geometry calculations.
        - Returns both the rectangle and the convex hull in WGS84 coordinates.
    """
    _validate_polygon(polygon)

    try:
        wgs84_to_utm, utm_to_wgs84 = get_utm_transforms(polygon)

        # Transform to UTM and calculate convex hull and minimum rotated rectangle
        polygon_utm = transform(wgs84_to_utm, polygon).convex_hull
        mrr = polygon_utm.minimum_rotated_rectangle

        # Transform results back to WGS84
        mrr_wgs84 = transform(utm_to_wgs84, mrr)

    except Exception as e:
        raise ValueError(f"Failed to calculate minimum rotated rectangle: {e}")

    return mrr_wgs84


def rotated_rectangle(polygon: Polygon, azimuth: float) -> Polygon:
    """
    Compute a rotated bounding rectangle around a Shapely polygon in WGS84 coordinates at a specified azimuth.

    Args:
        polygon (Polygon): Input polygon in WGS84 coordinates. Must be valid.
        azimuth (float): Desired azimuth for the bounding rectangle in degrees. Will be wrapped to [-180, 180].

    Returns:
        Polygon: The rotated bounding rectangle in WGS84 coordinates.

    Raises:
        ValueError: If the input polygon is invalid or if an error occurs during processing.

    Notes:
        - The input polygon is transformed to UTM for accurate geometry calculations.
        - The bounding rectangle is rotated to align with the specified azimuth.
        - The result is returned in WGS84 coordinates.
    """
    # Validate inputs
    _validate_polygon(polygon)

    try:
        # Transform polygon to UTM
        wgs84_to_utm, utm_to_wgs84 = get_utm_transforms(polygon)
        polygon_utm = transform(wgs84_to_utm, polygon).convex_hull

        # Perform rotation
        azimuth_radians = np.radians(azimuth)
        cx, cy = polygon_utm.centroid.x, polygon_utm.centroid.y
        xt, yt = polygon_utm.exterior.xy
        xt, yt = np.array(xt) - cx, np.array(yt) - cy

        # Rotate coordinates
        xr = xt * np.cos(azimuth_radians) - yt * np.sin(azimuth_radians)
        yr = xt * np.sin(azimuth_radians) + yt * np.cos(azimuth_radians)

        # Compute bounding box in rotated space
        minx_r, miny_r, maxx_r, maxy_r = np.min(xr), np.min(yr), np.max(xr), np.max(yr)
        xbound_r = np.array([minx_r, minx_r, maxx_r, maxx_r, minx_r])
        ybound_r = np.array([miny_r, maxy_r, maxy_r, miny_r, miny_r])

        # Rotate bounding box back to original space
        xbound = (xbound_r * np.cos(-azimuth_radians) - ybound_r * np.sin(-azimuth_radians)) + cx
        ybound = (xbound_r * np.sin(-azimuth_radians) + ybound_r * np.cos(-azimuth_radians)) + cy

        # Create rotated bounding box
        rotated_bbox_utm = Polygon(zip(xbound, ybound))
        rotated_bbox_wgs84 = transform(utm_to_wgs84, rotated_bbox_utm)

    except Exception as e:
        raise ValueError(f"Failed to compute rotated bounding rectangle: {e}")

    return rotated_bbox_wgs84



def translate_polygon(polygon: Polygon, distance: float, azimuth: float) -> Polygon:
    """
    Translate a Shapely polygon by a specified distance in a given rotational direction.

    Args:
        polygon (Polygon): The input Shapely polygon to be translated
        distance (float): Distance to translate the polygon (in the same units as the polygon's coordinates).
        azimuth (float): Angle of translation in degrees, measured clockwise from north.

    Returns:
        Polygon: The translated Shapely polygon.
    """

    # Convert the angle to radians
    azimuth_radians = np.radians(azimuth)

    # Calculate the x and y offsets
    x_offset = distance * np.sin(azimuth_radians)
    y_offset = distance * np.cos(azimuth_radians)

    # Translate the polygon
    translated_polygon = translate(polygon, xoff=x_offset, yoff=y_offset)

    return translated_polygon


def buffer_polygon_along_azimuth(polygon: Polygon, along_track_distance: float, across_track_distance: float, azimuth: float) -> Polygon:
    """
    Translate a Shapely polygon in both a specified direction and its opposite,
    then compute the convex hull of the union of the two translated polygons.

    Args:
        polygon (Polygon): The input Shapely polygon to be buffered in WGS84 coordinates. Must be valid.
        distance (ureg.Quantity): Distance to translate the polygon. Must be a positive length Quantity.
        azimuth (float): Angle of translation in degrees, measured clockwise from north.

    Returns:
        Polygon: The convex hull of the union of the two translated polygons in WGS84 coordinates.

    Raises:
        ValueError: If the input polygon is invalid or if distance is not a valid length.

    Notes:
        - The input polygon is transformed to UTM for accurate geometry calculations.
        - The resulting convex hull is returned in WGS84 coordinates.
    """
    # Validate inputs
    _validate_polygon(polygon)

    for key, value in {'along_track_distance': along_track_distance, 'across_track_distance': across_track_distance}.items():
        if not isinstance(value, float):
            raise ValueError(f"Invalid type for '{key}': Expected float (meters) or ureg.Quantity. Got {type(value)}.")
        if value <= 0:
            raise ValueError("Distance must be greater than 0.")

    azimuth = wrap_to_180(azimuth)

    try:
        # Transform to UTM
        wgs84_to_utm, utm_to_wgs84 = get_utm_transforms(polygon)
        polygon_utm = transform(wgs84_to_utm, polygon)

        # Translate in both directions
        translated_polygon_1 = translate_polygon(polygon_utm, across_track_distance, azimuth+90)
        translated_polygon_2 = translate_polygon(polygon_utm, across_track_distance, azimuth-90)

        # Compute convex hull of the union
        polygon_utm = unary_union([translated_polygon_1, translated_polygon_2])

        translated_polygon_1 = translate_polygon(polygon_utm, along_track_distance, azimuth)
        translated_polygon_2 = translate_polygon(polygon_utm, along_track_distance, azimuth-180)

        polygon_utm = unary_union([translated_polygon_1, translated_polygon_2])

        buffered_polygon_wgs84 = transform(utm_to_wgs84, polygon_utm)

    except Exception as e:
        raise ValueError(f"Failed to buffer polygon along azimuth: {e}")

    return buffered_polygon_wgs84
