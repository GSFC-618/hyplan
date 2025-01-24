import pandas as pd
import numpy as np

import math
from shapely.geometry import Point, LineString
from shapely.ops import transform
from typing import Union
from pymap3d.lox import meanm

from .geometry import get_utm_transforms
from .units import ureg

class Waypoint:
    def __init__(self, latitude: float, longitude: float, heading: float, altitude: Union[ureg.Quantity, float, None] = None):
        """
        Initialize a Waypoint object.

        Args:
            latitude (float): Latitude in decimal degrees.
            longitude (float): Longitude in decimal degrees.
            heading (float): Heading in degrees relative to North.
            altitude (Union[Quantity, float, None], optional): Altitude in meters or as a pint Quantity. Defaults to None.
        """
        # Validate latitude and longitude
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError("Longitude must be between -180 and 180 degrees")

        # Validate and process altitude
        if altitude is None:
            self.altitude = None
        elif isinstance(altitude, float):
            self.altitude = altitude  # Assume meters if a float is provided
        elif hasattr(altitude, 'units') and altitude.check('[length]'):
            self.altitude = altitude.to(ureg.meter).magnitude
        else:
            raise TypeError("Altitude must be None, a float (meters), or a pint Quantity with length units")

        self.point = Point(longitude, latitude)
        self.heading = heading

class DubinsPath:
    def __init__(self, start: Waypoint, end: Waypoint, speed: Union[ureg.Quantity, float], bank_angle: float, step_size: float):
        """
        Initialize a DubinsPath object.

        Args:
            start (Waypoint): The starting waypoint.
            end (Waypoint): The ending waypoint.
            speed (Union[Quantity, float]): Speed as a pint Quantity or a float (meters per second).
            bank_angle (float): Bank angle in degrees.
            step_size (float): Step size for sampling the trajectory.
        """
        if not isinstance(start, Waypoint) or not isinstance(end, Waypoint):
            raise TypeError("start and end must be Waypoint objects")

        self.start = start
        self.end = end

        if isinstance(speed, float):
            self.speed_mps = speed  # Assume meters per second if speed is a float
        elif hasattr(speed, 'units') and speed.check('[speed]'):
            self.speed_mps = speed.to(ureg.meter / ureg.second).magnitude
        else:
            raise TypeError("speed must be a pint Quantity with speed units or a float (meters per second)")

        self.bank_angle = bank_angle
        self.step_size = step_size

        # Calculate Dubins path properties
        self._geometry = None
        self._length = None
        self._calculate_path()

    def _calculate_path(self):
        """
        Calculate the Dubins path and its properties.
        """
        # Convert bank angle to radians
        bank_angle_rad = math.radians(self.bank_angle)

        # Calculate the turn radius
        g = 9.8  # m/s^2
        turn_radius = (self.speed_mps ** 2) / (g * math.tan(bank_angle_rad))

        # Convert azimuths to radians
        heading1 = math.radians(self.start.heading)
        heading2 = math.radians(self.end.heading)

        # Calculate the geographic mean (midpoint)
        midpoint_lat, midpoint_lon = meanm([self.start.point.y, self.end.point.y], [self.start.point.x, self.end.point.x])

        # Get UTM transforms
        to_utm, from_utm = get_utm_transforms(midpoint_lat, midpoint_lon)

        # Transform points to UTM
        start_utm = transform(to_utm, self.start.point)
        end_utm = transform(to_utm, self.end.point)

        # Define the start and end configurations
        q0 = (start_utm.x, start_utm.y, heading1)
        q1 = (end_utm.x, end_utm.y, heading2)

        # Generate the Dubins path
        qs, _ = dubins.path_sample(q0, q1, turn_radius, self.step_size)

        # Convert sampled points back to geographic coordinates
        dubins_path_coords = [
            (transform(from_utm, Point(x, y)).y, transform(from_utm, Point(x, y)).x) for x, y, _ in qs
        ]

        # Create LineString and calculate length
        self._geometry = LineString([(lon, lat) for lat, lon in dubins_path_coords])
        self._length = self._calculate_length(qs)
    
    def _calculate_length(self, qs) -> float:
        """
        Calculate the length of the Dubins path directly from the sampled points using vectorized operations.
    
        Args:
            qs (list): List of sampled points [(x1, y1, h1), (x2, y2, h2), ...].
    
        Returns:
            float: Total length of the path in meters.
        """
        coordinates = np.array([(x, y) for x, y, _ in qs])
        diffs = np.diff(coordinates, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        return distances.sum()

    @property
    def geometry(self) -> LineString:
        """
        Get the Dubins path as a LineString.

        Returns:
            LineString: The Dubins path.
        """
        return self._geometry

    @property
    def length(self) -> float:
        """
        Get the length of the Dubins path in meters.

        Returns:
            float: Length of the path in meters.
        """
        return self._length
