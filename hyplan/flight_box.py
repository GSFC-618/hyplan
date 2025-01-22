import numpy as np
import pymap3d.vincenty
from typing import Optional, List, Callable, Dict, Union
from shapely.geometry import Polygon
import logging

from . import flight_line
from .units import ureg, altitude_to_flight_level
from .geometry import wrap_to_180, rotated_rectangle, minimum_rotated_rectangle, buffer_polygon_along_azimuth, _validate_polygon


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _validate_inputs(**kwargs) -> None:
    """
    Validate input parameters for various operations using dynamic rules.

    Args:
        **kwargs: Arbitrary keyword arguments representing parameters to validate.
            Supported parameters and their rules:
            - altitude: Must be a positive float (meters) or a `ureg.Quantity` with length dimensionality.
            - box_length: Must be a positive float (meters) or a `ureg.Quantity` with length dimensionality.
            - box_width: Must be a positive float (meters) or a `ureg.Quantity` with length dimensionality.
            - overlap: Must be a float between 0 and 100 (inclusive).
            - starting_point: Must be either "edge" or "center".
            - azimuth: Must be a float, wrapped to [-180, 180] degrees.
            - polygon: If provided, must be a valid Shapely Polygon.
            - clip_to_polygon: Must be a boolean.

    Raises:
        ValueError: If any parameter fails its validation rule.

    Notes:
        - Length-related parameters (`altitude`, `box_length`, `box_width`) are checked for dimensionality if they are `ureg.Quantity` and converted to meters.
        - Unknown parameters will be ignored, with a warning logged.
    """
    rules: Dict[str, Callable[[Union[float, Polygon, bool, None]], bool]] = {
        'altitude': lambda x: isinstance(x, (float, ureg.Quantity)) and x > 0,
        'box_length': lambda x: isinstance(x, (float, ureg.Quantity)) and x > 0,
        'box_width': lambda x: isinstance(x, (float, ureg.Quantity)) and x > 0,
        'overlap': lambda x: isinstance(x, (float, int)) and 0 <= x <= 100,
        'starting_point': lambda x: x in {"edge", "center"},
        'azimuth': lambda x: isinstance(x, float),
        'polygon': lambda x: x is None or _validate_polygon(x),
        'starting_point': lambda x: x in {"edge", "center"},
        'clip_to_polygon': lambda x: isinstance(x, bool),
    }

    for key, value in kwargs.items():
        if key in {'altitude', 'box_length', 'box_width'}:
            # Validate and process length-related parameters
            if isinstance(value, ureg.Quantity):
                if not value.check("[length]"):
                    raise ValueError(f"Invalid unit for '{key}': Expected a length unit. Got {value.dimensionality}.")
                value = value.to("meter").magnitude  # Convert to meters
            elif not isinstance(value, float):
                raise ValueError(f"Invalid type for '{key}': Expected float (meters) or ureg.Quantity. Got {type(value)}.")
            
            if value <= 0:
                raise ValueError(f"Invalid value for '{key}': {value}. Must be greater than 0.")

        elif key == 'azimuth':
            # Validate and wrap azimuth
            if not isinstance(value, float):
                raise ValueError(f"Invalid type for 'azimuth': Expected float. Got {type(value)}.")
            kwargs[key] = wrap_to_180(value)

        # elif key == 'polygon' and value is not None:
        #     # Validate polygon
        #     _validate_polygon(value)

        elif key in rules:
            # Validate other parameters using rules
            if not rules[key](value):
                raise ValueError(f"Invalid value for '{key}': {value}. Check documentation for valid inputs.")
        else:
            # Warn about unknown parameters
            logging.warning(f"Unknown parameter '{key}' provided. No validation rule exists.")
    
    logging.debug("All inputs passed validation.")

        
def box_around_center_line(
    instrument: object,
    altitude: ureg.Quantity,
    lat0: float,
    lon0: float,
    azimuth: float,
    box_length: ureg.Quantity,
    box_width: ureg.Quantity,
    box_name: str = "Line",
    start_numbering: int = 1,
    overlap: float = 20,
    alternate_direction: bool = True,
    starting_point: str = "center",
    polygon: Optional[Polygon] = None,
) -> List[flight_line.FlightLine]:
    """
    Create a series of flight lines around a center line based on the given box dimensions and instrument properties.

    Args:
        instrument (object): An object with a method `swath_width_at(altitude: Quantity) -> Quantity`
            that returns the swath width at a given altitude.
        altitude (ureg.Quantity): Altitude for the flight lines. Must be a positive length Quantity.
        lat0 (float): Latitude of the box center in decimal degrees (-90 to 90).
        lon0 (float): Longitude of the box center in decimal degrees (-180 to 180).
        azimuth (float): Orientation of the box in degrees. Will be wrapped to [-180, 180].
        box_length (ureg.Quantity): Length of the box as a positive length Quantity.
        box_width (ureg.Quantity): Width of the box as a positive length Quantity.
        box_name (str): Name prefix for the flight lines.
        start_numbering (int): Starting number for flight line naming. Must be a positive integer.
        overlap (float): Percentage overlap between adjacent swaths. Must be between 0 and 100.
        alternate_direction (bool): Whether to alternate flight line directions.
        starting_point (str): Whether to start the first line from the "edge" or "center" of the box.
        polygon (Optional[Polygon]): Optional polygon to clip flight lines. Must be valid.

    Returns:
        List[flight_line.FlightLine]: A list of generated flight lines.

    Raises:
        ValueError: If inputs do not meet validation criteria.

    Notes:
        - Flight lines are generated around the center of the box, with distances calculated from the centerline.
        - Clipping is applied to each line if a polygon is provided.
    """
    # Validate inputs
    _validate_inputs(
        altitude=altitude,
        box_length=box_length,
        box_width=box_width,
        overlap=overlap,
        azimuth=azimuth,
        polygon=polygon
    )


    if not hasattr(instrument, "swath_width_at") or not callable(instrument.swath_width_at):
        raise ValueError("Instrument must have a callable method `swath_width_at(altitude)`.")

    # Compute swath spacing and number of lines
    swath = instrument.swath_width_at(altitude)
    if swath <= 0:
        raise ValueError(f"Invalid swath width {swath}. Must be positive.")

    swath_spacing = swath * (1 - (overlap / 100))
    if swath_spacing <= 0:
        raise ValueError(f"Invalid swath spacing {swath_spacing}. Adjust overlap or instrument parameters.")

    if polygon:
        along_track_buffer = 2000.0
        polygon = buffer_polygon_along_azimuth(polygon, along_track_buffer, swath.magnitude/2, azimuth)
        box_length += ureg.Quantity(along_track_buffer, "meter")

    nlines = max(1, int(np.ceil(box_width / swath_spacing)))

    logging.info(f"Calculated swath spacing: {swath_spacing:.2f} meters.")
    logging.info(f"Number of lines: {nlines}.")

    # Generate flight lines
    dists_from_center = np.arange(-nlines // 2, nlines // 2 + 1) * swath_spacing

    if starting_point == "edge":
        first_line = flight_line.FlightLine.start_length_azimuth(
            lat1=lat0, lon1=lon0, length=box_length, az=azimuth, altitude=altitude
        )
    elif starting_point == "center":
        first_line = flight_line.FlightLine.center_length_azimuth(
            lat=lat0, lon=lon0, length=box_length, az=azimuth, altitude=altitude
        )

    flight_level = altitude_to_flight_level(altitude)
    lines = []

    for idx, dist in enumerate(dists_from_center):
        line = first_line.offset_across(dist)
        if alternate_direction and idx % 2 == 0:
            line = line.reverse()

        line.site_name = f"{box_name}_L{idx + start_numbering:02d}_{flight_level}"

        if polygon:
            clipped_lines = line.clip_to_polygon(polygon)
            if clipped_lines:
                lines.extend(clipped_lines)
                logging.debug(f"Line {line.site_name} clipped into {len(clipped_lines)} segments.")
            else:
                logging.info(f"Line {line.site_name} fully excluded after clipping.")
        else:
            lines.append(line)

    if not lines:
        logging.warning("No flight lines were generated after clipping.")

    return lines


def box_around_minimum_rectangle(
    instrument: object,
    altitude: ureg.Quantity,
    polygon: Polygon,
    box_name: str = "Line",
    start_numbering: int = 1,
    overlap: float = 20,
    alternate_direction: bool = True,
    clip_to_polygon: bool = True,
) -> List[flight_line.FlightLine]:
    """
    Generate flight lines based on the minimum rotated rectangle of a polygon.

    Args:
        instrument (object): An object with a method `swath_width_at(altitude: Quantity) -> Quantity`
            that returns the swath width at a given altitude.
        altitude (ureg.Quantity): Altitude for the flight lines. Must be a positive length Quantity.
        polygon (Polygon): Input polygon to calculate the minimum rotated rectangle. Must be valid.
        box_name (str): Name prefix for the flight lines.
        start_numbering (int): Starting number for flight line naming. Must be a positive integer.
        overlap (float): Percentage overlap between adjacent swaths. Must be between 0 and 100.
        alternate_direction (bool): Whether to alternate flight line directions.
        clip_to_polygon (bool): Whether to clip flight lines to the convex hull of the polygon.

    Returns:
        List[flight_line.FlightLine]: A list of generated flight lines.

    Raises:
        ValueError: If inputs do not meet validation criteria.

    Notes:
        - The polygon's minimum rotated rectangle determines the box's center, azimuth, length, and width.
        - Flight lines are optionally clipped to the convex hull of the input polygon.
    """
    # Validate inputs
    _validate_inputs(
        altitude=altitude,
        overlap=overlap,
        polygon=polygon,
        clip_to_polygon=clip_to_polygon
    )

    if not isinstance(polygon, Polygon):
        raise ValueError("Input must be a Shapely Polygon.")

    # Calculate the minimum rotated rectangle
    try:
        mrr_wgs84 = minimum_rotated_rectangle(polygon)
    except Exception as e:
        raise ValueError(f"Failed to calculate minimum rotated rectangle for polygon: {e}")
    
    # Extract centroid
    lon0, lat0 = mrr_wgs84.centroid.coords[0]

    # Extract dimensions and azimuth
    lons, lats = list(mrr_wgs84.exterior.coords.xy)
    length1, az1 = pymap3d.vincenty.vdist(lats[0], lons[0], lats[3], lons[3])
    length2, az2 = pymap3d.vincenty.vdist(lats[0], lons[0], lats[1], lons[1])

    # Assign length, width, and azimuth
    if length1 <= length2:
        box_length, box_width, azimuth = length2, length1, wrap_to_180(az2)
    else:
        box_length, box_width, azimuth = length1, length2, wrap_to_180(az1)

    if box_length <= 0 or box_width <= 0:
        raise ValueError(f"Invalid box dimensions derived from polygon: Length={box_length}, Width={box_width}.")

    # Convert dimensions to pint quantities
    box_length = ureg.Quantity(box_length, "meter")
    box_width = ureg.Quantity(box_width, "meter")

    logging.info(
        f"Derived box parameters: Center=({lat0:.6f}, {lon0:.6f}), Azimuth={azimuth:.2f}°, "
        f"Length={box_length.magnitude:.2f} m, Width={box_width.magnitude:.2f} m."
    )

    # Generate flight lines using the box_around_center_line function
    return box_around_center_line(
        instrument=instrument,
        altitude=altitude,
        lat0=lat0,
        lon0=lon0,
        azimuth=azimuth,
        box_length=box_length,
        box_width=box_width,
        box_name=box_name,
        start_numbering=start_numbering,
        overlap=overlap,
        alternate_direction=alternate_direction,
        polygon=polygon if clip_to_polygon else None,
    )


def box_around_rotated_rectangle(
    instrument: object,
    altitude: ureg.Quantity,
    polygon: Polygon,
    azimuth: float,
    box_name: str = "RotatedBoxLine",
    start_numbering: int = 1,
    overlap: float = 20,
    alternate_direction: bool = True,
    clip_to_polygon: bool = True,
    starting_point: str = "center",  # Options: "edge" or "center"
) -> List[flight_line.FlightLine]:
    """
    Generate flight lines based on the rotated rectangle of a polygon with a specific azimuth.

    Args:
        instrument (object): An object with a method `swath_width_at(altitude: Quantity) -> Quantity`
            that returns the swath width at a given altitude.
        altitude (ureg.Quantity): Altitude for the flight lines. Must be a positive length Quantity.
        polygon (Polygon): Input polygon to calculate the rotated rectangle. Must be valid.
        azimuth (float): Desired azimuth for the bounding rectangle. Will be wrapped to [-180, 180].
        box_name (str): Name prefix for the flight lines.
        start_numbering (int): Starting number for flight line naming. Must be a positive integer.
        overlap (float): Percentage overlap between adjacent swaths. Must be between 0 and 100.
        alternate_direction (bool): Whether to alternate flight line directions.
        clip_to_polygon (bool): Whether to clip flight lines to the convex hull of the rotated rectangle.
        starting_point (str): Whether to start the first line from the "edge" or "center" of the box.

    Returns:
        List[flight_line.FlightLine]: A list of generated flight lines.

    Raises:
        ValueError: If inputs do not meet validation criteria.

    Notes:
        - The rotated rectangle is calculated using the specified azimuth.
        - Flight lines are optionally clipped to the bounding rectangle if `clip_to_polygon` is True.
    """
    # Validate inputs
    _validate_inputs(
        altitude=altitude,
        azimuth=azimuth,
        overlap=overlap,
        polygon=polygon,
        starting_point=starting_point,
        clip_to_polygon=clip_to_polygon
    )
    # Compute rotated rectangle
    try:
        rotated_bbox = rotated_rectangle(polygon, azimuth)
    except Exception as e:
        raise ValueError(f"Failed to calculate rotated rectangle for polygon: {e}")

    minx, miny, maxx, maxy = rotated_bbox.bounds
    box_length = pymap3d.vincenty.vdist(miny, minx, maxy, minx)[0]
    box_width = pymap3d.vincenty.vdist(miny, minx, miny, maxx)[0]

    if box_length <= 0 or box_width <= 0:
        raise ValueError(f"Invalid box dimensions derived from polygon: Length={box_length}, Width={box_width}.")

    # Convert dimensions to pint quantities
    box_length = ureg.Quantity(box_length, "meter")
    box_width = ureg.Quantity(box_width, "meter")

    logging.info(
        f"Rotated box parameters: Center=({rotated_bbox.centroid.y:.6f}, {rotated_bbox.centroid.x:.6f}), "
        f"Azimuth={azimuth:.2f}°, Length={box_length.magnitude:.2f} m, Width={box_width.magnitude:.2f} m."
    )

    # Generate flight lines using the box_around_center_line function
    return box_around_center_line(
        instrument=instrument,
        altitude=altitude,
        lat0=rotated_bbox.centroid.y,
        lon0=rotated_bbox.centroid.x,
        azimuth=azimuth,
        box_length=box_length,
        box_width=box_width,
        box_name=box_name,
        start_numbering=start_numbering,
        overlap=overlap,
        alternate_direction=alternate_direction,
        polygon=polygon if clip_to_polygon else None,
    )


