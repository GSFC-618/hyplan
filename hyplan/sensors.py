from typing import Dict
from pydantic import BaseModel
from pydantic_pint import PydanticPintQuantity
from pint import Quantity
import numpy as np
from typing import Annotated

from .units import ureg

class Sensor(BaseModel):
    """Base class to represent a generic sensor."""

    name: str

    def __str__(self) -> str:
        return self.name

class LineScanner(Sensor):
    """Class to represent a line scanning imager."""

    fov: Annotated[Quantity, PydanticPintQuantity("degree", ureg=ureg)]
    ifov: Annotated[Quantity, PydanticPintQuantity("degree", ureg=ureg)]
    across_track_pixels: int
    frame_rate: Annotated[Quantity, PydanticPintQuantity("Hz", ureg=ureg)]

    @property
    def half_angle(self) -> Annotated[Quantity, PydanticPintQuantity("degree")]:
        """Calculate and return the half angle."""
        return self.fov / 2.0

    @property
    def frame_period(self) -> Annotated[Quantity, PydanticPintQuantity("s")]:
        """Calculate and return the frame period."""
        return (1.0 / self.frame_rate).to(ureg.s)

    def swath_width_at(
        self, altitude: Annotated[Quantity, PydanticPintQuantity("m")]
    ) -> Annotated[Quantity, PydanticPintQuantity("m")]:
        """Calculate swath width (m) for a given altitude."""
        return 2 * altitude * np.tan(np.radians(self.fov / 2))

    def alt_for_pixel_size_nadir(
        self, px_size: Annotated[Quantity, PydanticPintQuantity("m")]
    ) -> Annotated[Quantity, PydanticPintQuantity("m")]:
        """Calculate altitude (m) for a given pixel size (m) at nadir."""
        return px_size / (2 * np.tan(np.radians(self.ifov / 2)))

    def alt_for_pixel_size(
        self, px_size: Annotated[Quantity, PydanticPintQuantity("m")]
    ) -> Annotated[Quantity, PydanticPintQuantity("m")]:
        """Calculate altitude (m) for an average pixel size (m)."""
        return (
            self.across_track_pixels * px_size / (2 * np.tan(np.radians(self.fov / 2)))
        )

    def pixel_size_at(
        self, altitude: Annotated[Quantity, PydanticPintQuantity("m")]
    ) -> Annotated[Quantity, PydanticPintQuantity("m")]:
        """Calculate average pixel size (m) for a given altitude (m)."""
        return self.swath_width_at(altitude) / self.across_track_pixels

    def pixel_size_at_nadir(
        self, altitude: Annotated[Quantity, PydanticPintQuantity("m")]
    ) -> Annotated[Quantity, PydanticPintQuantity("m")]:
        """Calculate pixel size (m) at nadir for a given altitude."""
        return 2 * altitude * np.tan(np.radians(self.ifov / 2))

    def critical_ground_speed(
        self,
        altitude: Annotated[Quantity, PydanticPintQuantity("m")],
        along_track_sampling: float = 1,
    ) -> Annotated[Quantity, PydanticPintQuantity("m/s")]:
        """Calculate the maximum ground speed (m/s) at which spectrometer can move to achieve a specified along track sampling rate."""
        return self.pixel_size_at_nadir(altitude) / (
            self.frame_period * along_track_sampling
        )


class FrameCamera(Sensor):
    """Class to represent a frame camera."""

    sensor_width: Annotated[Quantity, PydanticPintQuantity("mm", ureg=ureg)]
    sensor_height: Annotated[Quantity, PydanticPintQuantity("mm", ureg=ureg)]
    focal_length: Annotated[Quantity, PydanticPintQuantity("mm", ureg=ureg)]
    resolution_x: int
    resolution_y: int
    frame_rate: Annotated[Quantity, PydanticPintQuantity("Hz", ureg=ureg)]
    f_speed: float

    @property
    def fov_x(self) -> Annotated[Quantity, PydanticPintQuantity("degree")]:
        """Calculate horizontal Field of View (FoV)."""
        return 2 * np.degrees(np.arctan((self.sensor_width / (2 * self.focal_length)).magnitude))

    @property
    def fov_y(self) -> Annotated[Quantity, PydanticPintQuantity("degree")]:
        """Calculate vertical Field of View (FoV)."""
        return 2 * np.degrees(np.arctan((self.sensor_height / (2 * self.focal_length)).magnitude))

    def pixel_size_at(
        self, altitude: Annotated[Quantity, PydanticPintQuantity("m")]
    ) -> Dict[str, Annotated[Quantity, PydanticPintQuantity("m")]]:
        """Calculate pixel sizes (m) for a given altitude."""
        return {
            "x": (2 * altitude * np.tan(np.radians(self.fov_x / 2)) / self.resolution_x),
            "y": (2 * altitude * np.tan(np.radians(self.fov_y / 2)) / self.resolution_y),
        }

    def footprint_at(
        self, altitude: Annotated[Quantity, PydanticPintQuantity("m")]
    ) -> Dict[str, Annotated[Quantity, PydanticPintQuantity("m")]]:
        """Calculate the footprint dimensions (m) for a given altitude."""
        return {
            "width": 2 * altitude * np.tan(np.radians(self.fov_x / 2)),
            "height": 2 * altitude * np.tan(np.radians(self.fov_y / 2)),
        }

    def critical_ground_speed(
        self, altitude: Annotated[Quantity, PydanticPintQuantity("m")]
    ) -> Annotated[Quantity, PydanticPintQuantity("m/s")]:
        """
        Calculate the maximum ground speed (m/s) to maintain proper along-track sampling.

        Args:
            altitude (Quantity): Altitude of the camera in meters.

        Returns:
            Quantity: Maximum allowable ground speed in meters per second.
        """
        pixel_size = self.pixel_size_at(altitude)["y"]  # Along-track GSD
        frame_period = (1 / self.frame_rate).to(ureg.s)
        return pixel_size / frame_period



class AVIRISClassic(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="AVIRIS Classic",
            fov="34 degree",
            ifov="0.0572958 degree",
            across_track_pixels=677,
            frame_rate="100 Hz",
            **kwargs
        )


class AVIRISNextGen(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="AVIRIS Next Gen",
            fov="36.0 degree",
            ifov="0.0572958 degree",
            across_track_pixels=600,
            frame_rate="100 Hz",
            **kwargs
        )


class AVIRIS3(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="AVIRIS 3",
            fov="40.2 degree",
            ifov="0.031799 degree",
            across_track_pixels=1240,
            frame_rate="216 Hz",
            **kwargs
        )


class AVIRIS4(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="AVIRIS 4",
            fov="39.5 degree",
            ifov="0.03183869172 degree",
            across_track_pixels=1240,
            frame_rate="215 Hz",
            **kwargs
        )


class HyTES(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="HyTES",
            fov="50 degree",
            ifov="0.0977 degree",
            across_track_pixels=512,
            frame_rate="36 Hz",
            **kwargs
        )


class PRISM(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="PRISM",
            fov="30.7 degree",
            ifov="0.050534 degree",
            across_track_pixels=608,
            frame_rate="176 Hz",
            **kwargs
        )


class MASTER(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="MASTER",
            fov="85.92 degree",
            ifov="0.143239 degree",
            across_track_pixels=716,
            frame_rate="25 Hz",
            **kwargs
        )


class GLiHT_VNIR(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="G-LiHT VNIR",
            fov="64.0 degree",  # FOV in degrees
            ifov="0.03997984903 degree",  # IFOV in degrees
            across_track_pixels=1600,  # Across-track pixels
            frame_rate="250 Hz",  # Frame rate in Hz
            **kwargs
        )


class GLiHT_Thermal(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="G-LiHT Thermal",
            fov="42.6 degree",  # FOV in degrees
            ifov="0.054087216 degree",  # IFOV in degrees
            across_track_pixels=640,  # Across-track pixels
            frame_rate="50 Hz",  # Frame rate in Hz
            **kwargs
        )


class GLiHT_SIF(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="G-LiHT SIF",
            fov="23.5 degree",  # FOV in degrees
            ifov="0.1415206 degree",  # IFOV in degrees
            across_track_pixels=1600,  # Across-track pixels
            frame_rate="37.6 Hz",  # Frame rate in Hz
            **kwargs
        )


class CFIS(LineScanner):
    def __init__(self, **kwargs):
        super().__init__(
            name="CFIS",
            fov="11.46 degree",  # FOV in degrees
            ifov="0.04068 degree",  # IFOV in degrees
            across_track_pixels=256,  # Across-track pixels
            frame_rate="6 Hz",  # Frame rate in Hz
            **kwargs
        )


