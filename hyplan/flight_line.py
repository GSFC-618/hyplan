import logging
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString
from pint import Quantity
from typing import Optional, List, Dict, Union, Tuple
import pymap3d
import pymap3d.vincenty
import geopandas as gpd
import numpy as np

from .units import ureg
from .geometry import wrap_to_180
from .dubins_path import Waypoint

# Set up logging
logging.basicConfig(level=logging.INFO)

class FlightLine:
    """
    Represents a geospatial flight line with properties, validations, and operations.
    """
    def __init__(
        self,
        geometry: LineString,
        altitude: Quantity,
        site_name: Optional[str] = None,
        site_description: Optional[str] = None,
        investigator: Optional[str] = None,
    ):
        self._validate_geometry(geometry)
        self.geometry = geometry
        self.altitude = self._validate_altitude(altitude)
        self.site_name = site_name
        self.site_description = site_description
        self.investigator = investigator

    @staticmethod
    def _validate_geometry(geometry: LineString):
        if not isinstance(geometry, LineString):
            raise ValueError("Geometry must be a Shapely LineString.")
        if len(geometry.coords) != 2:
            raise ValueError("LineString must have exactly two points.")
        for lon, lat in geometry.coords:
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude {lat} is out of bounds (-90 to 90).")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude {lon} is out of bounds (-180 to 180).")

    @staticmethod
    def _validate_altitude(altitude: Quantity) -> Quantity:
        if not isinstance(altitude, Quantity):
            altitude = ureg.Quantity(altitude, "meter")
        else:
            altitude = altitude.to("meter")

        if altitude.magnitude < 0 or altitude.magnitude > 22000:
            logging.warning(
                f"Altitude {altitude.magnitude} meters is outside the typical range (0-22000 meters).")
        return altitude

    @property
    def lat1(self):
        return self.geometry.coords[0][1]

    @property
    def lon1(self):
        return self.geometry.coords[0][0]

    @property
    def lat2(self):
        return self.geometry.coords[-1][1]

    @property
    def lon2(self):
        return self.geometry.coords[-1][0]

    @property
    def length(self) -> Quantity:
        length, _ = pymap3d.vincenty.vdist(self.lat1, self.lon1, self.lat2, self.lon2)
        return ureg.Quantity(round(length,2), "meter")

    @property
    def az12(self) -> Quantity:
        _, az12 = pymap3d.vincenty.vdist(self.lat1, self.lon1, self.lat2, self.lon2)
        return ureg.Quantity(az12, "degree")

    @property
    def az21(self) -> Quantity:
        _, az21 = pymap3d.vincenty.vdist(self.lat2, self.lon2, self.lat1, self.lon1)
        return ureg.Quantity(az21, "degree")
    
    @property
    def waypoint1(self) -> Waypoint:
        return Waypoint(latitude=self.lat1, longitude=self.lon1, heading=self.az12.magnitude, altitude=self.altitude, name=self.site_name+"_start")
    
    @property
    def waypoint2(self) -> Waypoint:
        heading = (self.az21.magnitude + 180.0) % 360.0
        return Waypoint(latitude=self.lat2, longitude=self.lon2, heading=heading, altitude=self.altitude, name=self.site_name+"_end")
    
    @classmethod
    def start_length_azimuth(
        cls,
        lat1: float,
        lon1: float,
        length: Quantity,
        az: Quantity,
        **kwargs,
    ) -> "FlightLine":
        if not length.check("[length]"):
            raise ValueError("Length must have units of distance.")

        length_m = length.to("meter").magnitude
        lat2, lon2 = pymap3d.vincenty.vreckon(lat1, lon1, length_m, az)
        lon2 = wrap_to_180(lon2)

        geometry = LineString([(lon1, lat1), (lon2, lat2)])
        return cls(geometry=geometry, **kwargs)

    @classmethod
    def center_length_azimuth(
        cls,
        lat: float,
        lon: float,
        length: Quantity,
        az: Quantity,
        **kwargs,
    ) -> "FlightLine":
        if not length.check("[length]"):
            raise ValueError("Length must have units of distance.")

        length_m = length.to("meter").magnitude

        lat2, lon2 = pymap3d.vincenty.vreckon(lat, lon, length_m / 2, az)
        lat1, lon1 = pymap3d.vincenty.vreckon(lat, lon, length_m / 2, az - 180)

        lon1, lon2 = wrap_to_180(lon1), wrap_to_180(lon2)
        geometry = LineString([(lon1, lat1), (lon2, lat2)])
        return cls(geometry=geometry, **kwargs)

    def clip_to_polygon(
        self, clip_polygon: Union[Polygon, MultiPolygon]
    ) -> Optional[List["FlightLine"]]:
        """
        Clip the flight line to a specified polygon.

        Args:
            clip_polygon (Union[Polygon, MultiPolygon]): The polygon to clip the flight line to.

        Returns:
            Optional[List["FlightLine"]]: A list of resulting FlightLine(s), or None if no intersection exists.
        """
        clipped_geometry = self.geometry.intersection(clip_polygon)

        if clipped_geometry.is_empty:
            logging.info(f"FlightLine {self.site_name or '<Unnamed>'} excluded after clipping: No intersection.")
            return None

        if isinstance(clipped_geometry, LineString):
            if clipped_geometry.equals(self.geometry):
                logging.info(f"FlightLine {self.site_name or '<Unnamed>'} is entirely within the polygon.")
                return [self]  # No changes
            else:
                logging.info(f"FlightLine {self.site_name or '<Unnamed>'} was clipped into a single segment.")
                return [
                    FlightLine(
                        geometry=clipped_geometry,
                        altitude=self.altitude,
                        site_name=self.site_name,
                        site_description=self.site_description,
                        investigator=self.investigator,
                    )
                ]

        if isinstance(clipped_geometry, MultiLineString):
            results = []
            for i, segment in enumerate(clipped_geometry.geoms):
                new_site_name = f"{self.site_name}_{i:02d}" if self.site_name else f"Segment_{i:02d}"
                logging.info(f"FlightLine {self.site_name or '<Unnamed>'} was split into segment: {new_site_name}")
                results.append(
                    FlightLine(
                        geometry=segment,
                        altitude=self.altitude,
                        site_name=new_site_name,
                        site_description=self.site_description,
                        investigator=self.investigator,
                    )
                )
            return results

        logging.error(f"Unexpected geometry type after clipping: {type(clipped_geometry)}")
        raise TypeError(f"Unexpected geometry type after clipping: {type(clipped_geometry)}")


    def track(self, precision=100.0) -> LineString:
        """
        Generate a LineString representing the flight line.

        Args:
            precision (float): Desired distance (in meters) between interpolated points.

        Returns:
            LineString: A LineString object containing the interpolated track.
        """
        # Compute the number of points based on the length and precision
        num_points = int(np.ceil(self.length.to("meter").magnitude / precision)) + 1

        # Interpolate the points along the flight line
        track_lat, track_lon = pymap3d.vincenty.track2(
            self.lat1, self.lon1, self.lat2, self.lon2, npts=num_points, deg=True
        )

        # Wrap longitude to the range [-180, 180]
        track_lon = wrap_to_180(track_lon)

        # Create and return the LineString
        return LineString(zip(track_lon, track_lat))





    def reverse(self) -> "FlightLine":
        """
        Reverse the direction of the flight line.

        Returns:
            FlightLine: A new FlightLine object with reversed direction.
        """
        # Reverse the order of the coordinates
        reversed_geometry = LineString(list(reversed(self.geometry.coords)))

        # Create a new FlightLine with the reversed geometry
        return FlightLine(
            geometry=reversed_geometry,
            altitude=self.altitude,
            site_name=self.site_name,
            site_description=self.site_description,
            investigator=self.investigator
        )

    def offset_north_east(self, offset_north: Quantity, offset_east: Quantity) -> "FlightLine":
        """
        Offset the flight line in the north and east directions.

        Args:
            offset_north (Quantity): Distance to offset in the north direction (positive or negative).
            offset_east (Quantity): Distance to offset in the east direction (positive or negative).

        Returns:
            FlightLine: A new FlightLine object with the offset applied.
        """
        if not isinstance(offset_north, Quantity):
            offset_north = ureg.Quantity(offset_north, "meter")
        if not isinstance(offset_east, Quantity):
            offset_east = ureg.Quantity(offset_east, "meter")

        offset_north_m = offset_north.to("meter").magnitude
        offset_east_m = offset_east.to("meter").magnitude

        def compute_offset(lat, lon, north, east):
            new_lat, new_lon, _ = pymap3d.ned2geodetic(
                north, east, 0, lat, lon, self.altitude.magnitude
            )
            return new_lat, wrap_to_180(new_lon)

        new_lat1, new_lon1 = compute_offset(self.lat1, self.lon1, offset_north_m, offset_east_m)
        new_lat2, new_lon2 = compute_offset(self.lat2, self.lon2, offset_north_m, offset_east_m)

        new_lat1, new_lon1 = round(new_lat1, 6), round(new_lon1, 6)
        new_lat2, new_lon2 = round(new_lat2, 6), round(new_lon2, 6)

        offset_geometry = LineString([(new_lon1, new_lat1), (new_lon2, new_lat2)])

        return FlightLine(
            geometry=offset_geometry,
            altitude=self.altitude,
            site_name=self.site_name,
            site_description=self.site_description,
            investigator=self.investigator
        )

    def offset_across(self, offset_distance: Quantity) -> "FlightLine":
        """
        Offset the flight line perpendicular to its direction by a specified distance.

        Args:
            offset_distance (Quantity): Distance to offset the line (positive for right, negative for left).

        Returns:
            FlightLine: A new FlightLine object with the offset applied.
        """
        perpendicular_az = (self.az12.magnitude + 90) % 360 if offset_distance.magnitude >= 0 else (self.az12.magnitude - 90) % 360

        def compute_offset(lat, lon, distance, azimuth):
            return pymap3d.vincenty.vreckon(lat, lon, distance.to("meter").magnitude, azimuth)

        new_lat1, new_lon1 = compute_offset(self.lat1, self.lon1, abs(offset_distance), perpendicular_az)
        new_lat2, new_lon2 = compute_offset(self.lat2, self.lon2, abs(offset_distance), perpendicular_az)

        new_lon1, new_lon2 = wrap_to_180(new_lon1), wrap_to_180(new_lon2)
        new_lat1, new_lon1 = round(new_lat1, 6), round(new_lon1, 6)
        new_lat2, new_lon2 = round(new_lat2, 6), round(new_lon2, 6)

        offset_geometry = LineString([(new_lon1, new_lat1), (new_lon2, new_lat2)])

        return FlightLine(
            geometry=offset_geometry,
            altitude=self.altitude,
            site_name=self.site_name,
            site_description=self.site_description,
            investigator=self.investigator
        )

    def offset_along(self, offset_start: Quantity, offset_end: Quantity) -> "FlightLine":
        """
        Offset the flight line along its direction by modifying the start and end points.

        Args:
            offset_start (Quantity): Distance to offset the start point along the line (positive or negative).
            offset_end (Quantity): Distance to offset the end point along the line (positive or negative).

        Returns:
            FlightLine: A new FlightLine object with the offset applied.
        """
        def compute_offset(lat, lon, offset, azimuth):
            if offset < 0:
                azimuth = (azimuth + 180) % 360
                offset = abs(offset)
            return pymap3d.vincenty.vreckon(lat, lon, offset.to("meter").magnitude, azimuth)

        new_lat1, new_lon1 = compute_offset(self.lat1, self.lon1, offset_start, self.az12.magnitude)
        new_lat2, new_lon2 = compute_offset(self.lat2, self.lon2, offset_end, self.az21.magnitude)

        new_lon1, new_lon2 = wrap_to_180(new_lon1), wrap_to_180(new_lon2)
        new_lat1, new_lon1 = round(new_lat1, 6), round(new_lon1, 6)
        new_lat2, new_lon2 = round(new_lat2, 6), round(new_lon2, 6)

        offset_geometry = LineString([(new_lon1, new_lat1), (new_lon2, new_lat2)])

        return FlightLine(
            geometry=offset_geometry,
            altitude=self.altitude,
            site_name=self.site_name,
            site_description=self.site_description,
            investigator=self.investigator
        )

    def rotate_around_midpoint(self, angle: float) -> "FlightLine":
        """
        Rotate the flight line around its midpoint by a specified angle.

        Args:
            angle (float): Rotation angle in degrees. Positive values indicate counterclockwise rotation.

        Returns:
            FlightLine: A new FlightLine object rotated around its midpoint.
        """
        try:
            # Validate input
            if not isinstance(angle, (int, float)):
                raise ValueError(f"Angle must be a number. Received: {angle}")

            # Convert angle to radians
            angle_rad = np.radians(angle)

            # Compute the midpoint of the flight line
            midpoint = self.geometry.interpolate(0.5, normalized=True)

            # Helper function to rotate a single point
            def rotate_point(x, y, center_x, center_y, angle_radians):
                delta_x = x - center_x
                delta_y = y - center_y
                rotated_x = delta_x * np.cos(angle_radians) - delta_y * np.sin(angle_radians) + center_x
                rotated_y = delta_x * np.sin(angle_radians) + delta_y * np.cos(angle_radians) + center_y
                return rotated_x, rotated_y

            # Rotate both endpoints of the line
            rotated_coords = [
                rotate_point(x, y, midpoint.x, midpoint.y, angle_rad)
                for x, y in self.geometry.coords
            ]

            # Create the new rotated geometry
            rotated_geometry = LineString(rotated_coords)

            # Return a new FlightLine object with the rotated geometry
            return FlightLine(
                geometry=rotated_geometry,
                altitude=self.altitude,
                site_name=self.site_name,
                site_description=self.site_description,
                investigator=self.investigator,
            )

        except Exception as e:
            logging.error(f"Failed to rotate FlightLine: {e}")
            raise

    def split_by_length(self, max_length: Quantity, gap_length: Optional[Quantity] = None) -> List["FlightLine"]:
        """
        Split the flight line into segments of a specified maximum length with an optional gap between segments.

        Args:
            max_length (Quantity): Maximum length of each segment (meters).
            gap_length (Optional[Quantity]): Length of the gap between segments (meters). Default is None.

        Returns:
            List[FlightLine]: List of FlightLine objects representing the segments.
        """
        total_length_m = self.length.to("meter").magnitude
        max_length_m = max_length.to("meter").magnitude
        gap_length_m = gap_length.to("meter").magnitude if gap_length else 0

        if max_length_m <= 0:
            raise ValueError("Maximum length must be greater than 0.")
        if gap_length and gap_length_m < 0:
            raise ValueError("Gap length cannot be negative.")
        if max_length_m + gap_length_m > total_length_m:
            return [self]  # No split is possible if the segment + gap is longer than the flight line

        segments = []
        remaining_length_m = total_length_m
        current_start_lat, current_start_lon = self.lat1, self.lon1
        segment_index = 0

        while remaining_length_m > 0:
            # Calculate segment length, ensuring it doesn't exceed remaining length
            current_segment_length_m = min(max_length_m, remaining_length_m)
            remaining_length_m -= current_segment_length_m

            # Calculate endpoint for the current segment
            end_lat, end_lon = pymap3d.vincenty.vreckon(
                current_start_lat, current_start_lon, current_segment_length_m, self.az12.magnitude
            )
            end_lon = wrap_to_180(end_lon)

            # Create the segment geometry
            segment_geometry = LineString([(current_start_lon, current_start_lat), (end_lon, end_lat)])
            segments.append(
                FlightLine(
                    geometry=segment_geometry,
                    altitude=self.altitude,
                    site_name=f"{self.site_name}_seg_{segment_index}" if self.site_name else f"Segment_{segment_index}",
                    site_description=self.site_description,
                    investigator=self.investigator,
                )
            )
            segment_index += 1

            # Calculate the starting point for the next segment, skipping the gap
            if gap_length and remaining_length_m > gap_length_m:
                remaining_length_m -= gap_length_m
                current_start_lat, current_start_lon = pymap3d.vincenty.vreckon(
                    end_lat, end_lon, gap_length_m, self.az12.magnitude
                )
                current_start_lon = wrap_to_180(current_start_lon)
            else:
                break  # If no room for a new segment after the gap, stop splitting

        return segments

    def to_dict(self) -> Dict:
        return {
            "geometry": list(self.geometry.coords),
            "lat1": self.lat1,
            "lon1": self.lon1,
            "lat2": self.lat2,
            "lon2": self.lon2,
            "length": self.length.magnitude,
            "altitude_m": self.altitude.magnitude,
            "site_name": self.site_name,
            "site_description": self.site_description,
            "investigator": self.investigator,
        }

    def to_geojson(self) -> Dict:
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": list(self.geometry.coords),
            },
            "properties": {
                "altitude_m": self.altitude.magnitude,
                "site_name": self.site_name,
                "site_description": self.site_description,
                "investigator": self.investigator,
            },
        }


def to_gdf(flight_lines: List[FlightLine], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    data = [fl.to_dict() for fl in flight_lines]
    geometries = [fl.geometry for fl in flight_lines]
    return gpd.GeoDataFrame(data, geometry=geometries, crs=crs)
