import numpy as np
import logging
from .units import ureg
from .airports import Airport
from .dubins_path import Waypoint, DubinsPath

class Aircraft:
    def __init__(
        self, type, tail_number, service_ceiling, approach_speed, best_rate_of_climb,
        cruise_speed, range, endurance, operator, useful_payload, vx, vy,
        roc_at_service_ceiling=100 * ureg.feet / ureg.minute,
        descent_rate=1500 * ureg.feet / ureg.minute
    ):
        self.type = type
        self.tail_number = tail_number
        self.service_ceiling = service_ceiling
        self.approach_speed = approach_speed
        self.best_rate_of_climb = best_rate_of_climb
        self.cruise_speed = cruise_speed
        self.range = range.to(ureg.nautical_mile)
        self.endurance = endurance.to(ureg.hour)
        self.operator = operator
        self.useful_payload = useful_payload.to(ureg.pound)
        self.descent_rate = descent_rate
        self.vx = vx.to(ureg.knot)
        self.vy = vy.to(ureg.knot)
        self.roc_at_service_ceiling = roc_at_service_ceiling

    def rate_of_climb(self, altitude):
        """
        Compute the rate of climb at a given altitude assuming a linear decrease
        from best rate of climb at sea level to roc_at_service_ceiling at service ceiling.
        """
        if altitude >= self.service_ceiling:
            return self.roc_at_service_ceiling
        
        altitude_ratio = altitude / self.service_ceiling
        roc = (1 - altitude_ratio) * (self.best_rate_of_climb - self.roc_at_service_ceiling) + self.roc_at_service_ceiling
        return roc

    def time_to_climb(self, start_altitude, end_altitude, true_air_speed=None):
        """
        Estimate the time and horizontal distance traveled during a Vy climb.
        """
        try:
            start_altitude = start_altitude.to(ureg.feet)
            end_altitude = end_altitude.to(ureg.feet)
            true_air_speed = (true_air_speed or self.vy).to(ureg.feet / ureg.minute)

            if end_altitude > self.service_ceiling:
                raise ValueError("End altitude cannot exceed the service ceiling.")
            if end_altitude <= start_altitude:
                return 0 * ureg.minute, 0 * ureg.nautical_mile

            altitude_difference = end_altitude - start_altitude
            avg_rate_of_climb = (self.rate_of_climb(start_altitude) + self.rate_of_climb(end_altitude)) / 2
            time_to_climb = (altitude_difference / avg_rate_of_climb).to(ureg.minute)

            climb_angle = np.arctan(avg_rate_of_climb / true_air_speed).to(ureg.radian)
            horizontal_speed = (true_air_speed * np.cos(climb_angle)).to(ureg.nautical_mile / ureg.hour)
            horizontal_distance = (horizontal_speed * time_to_climb).to(ureg.nautical_mile)

            return time_to_climb, horizontal_distance

        except Exception as e:
            logging.error(f"Error in time_to_climb: {e}")
            raise

    def time_to_takeoff(self, airport: Airport, waypoint: Waypoint):
        """
        Calculate the total time needed to take off from an airport and reach a waypoint at cruise altitude,
        with detailed altitude and time information for each phase.
        """
        try:
            airport_altitude = airport.elevation.to(ureg.feet)
            waypoint_altitude = waypoint.altitude.to(ureg.feet)

            # Climb phase
            climb_time, climb_distance = self.time_to_climb(airport_altitude, waypoint_altitude)

            # Cruise phase
            dubins_path = DubinsPath(
                start=Waypoint(latitude=airport.latitude, longitude=airport.longitude, heading=0, altitude=airport_altitude),
                end=waypoint,
                speed=self.cruise_speed,
                bank_angle=25,
                step_size=100
            )
            total_distance = dubins_path.length.to(ureg.nautical_mile)
            cruise_distance = max(0 * ureg.nautical_mile, total_distance - climb_distance)
            cruise_time = (cruise_distance / self.cruise_speed).to(ureg.minute)

            total_time = climb_time + cruise_time

            # Return detailed phase information
            return {
                "total_time": total_time,
                "phases": {
                    "takeoff_climb": {
                        "start_altitude": airport_altitude,
                        "end_altitude": waypoint_altitude,
                        "start_time": 0 * ureg.minute,
                        "end_time": climb_time,
                    },
                    "takeoff_cruise": {
                        "start_altitude": waypoint_altitude,
                        "end_altitude": waypoint_altitude,
                        "start_time": climb_time,
                        "end_time": total_time,
                    },
                },
                "dubins_path": dubins_path
}


        except Exception as e:
            logging.error(f"Error in time_to_takeoff: {e}")
            raise


    def time_to_return(self, waypoint: Waypoint, airport: Airport):
        """
        Calculate the total time needed to return to an airport during an IFR landing,
        with detailed altitude and time information for each phase.
        """
        try:
            airport_waypoint = Waypoint(latitude=airport.latitude, longitude=airport.longitude, heading=waypoint.heading+90.0, altitude=airport.elevation)
            dubins_path = DubinsPath(
                start=waypoint,
                end=airport_waypoint,
                speed=self.cruise_speed,
                bank_angle=25,
                step_size=100
            )
            total_distance = dubins_path.length.to(ureg.nautical_mile)

            # Cruise phase
            cruise_altitude = waypoint.altitude.to(ureg.feet)
            approach_altitude = airport.elevation.to(ureg.feet) + 5_000 * ureg.feet
            descent_altitude = cruise_altitude - approach_altitude
            descent_distance = 3 * (descent_altitude.to(ureg.feet).magnitude / 1_000) * ureg.nautical_mile
            cruise_distance = max(0 * ureg.nautical_mile, total_distance - descent_distance)
            cruise_time = (cruise_distance / self.cruise_speed).to(ureg.minute)

            # Descent phase
            descent_time, descent_distance_actual = self.time_to_descend(cruise_altitude, approach_altitude)

            # Approach phase
            if_to_faf_distance = 16 * ureg.nautical_mile
            faf_to_runway_distance = 6 * ureg.nautical_mile
            average_speed_if_to_faf = (self.cruise_speed + self.approach_speed) / 2
            if_to_faf_time = (if_to_faf_distance / average_speed_if_to_faf).to(ureg.minute)
            faf_to_runway_time = (faf_to_runway_distance / self.approach_speed).to(ureg.minute)
            approach_time = if_to_faf_time + faf_to_runway_time

            total_time = cruise_time + descent_time + approach_time

            # Return detailed phase information
            return {
                "total_time": total_time,
                "phases": {
                    "return_cruise": {
                        "start_altitude": cruise_altitude,
                        "end_altitude": cruise_altitude,
                        "start_time": 0 * ureg.minute,
                        "end_time": cruise_time,
                    },
                    "return_descent": {
                        "start_altitude": cruise_altitude,
                        "end_altitude": approach_altitude,
                        "start_time": cruise_time,
                        "end_time": cruise_time + descent_time,
                    },
                    "return_approach": {
                        "start_altitude": approach_altitude,
                        "end_altitude": airport.elevation.to(ureg.feet),
                        "start_time": cruise_time + descent_time,
                        "end_time": total_time,
                    },
                },
                "dubins_path": dubins_path
            }


        except Exception as e:
            logging.error(f"Error in time_to_return: {e}")
            raise

    def time_to_descend(self, start_altitude, end_altitude, true_air_speed=None):
        """
        Estimate the time and horizontal distance traveled during descent.
        """
        try:
            start_altitude = start_altitude.to(ureg.feet)
            end_altitude = end_altitude.to(ureg.feet)
            true_air_speed = (true_air_speed or self.cruise_speed).to(ureg.feet / ureg.minute)

            if start_altitude <= end_altitude:
                return 0 * ureg.minute, 0 * ureg.nautical_mile

            altitude_difference = start_altitude - end_altitude
            time_to_descend = (altitude_difference / self.descent_rate).to(ureg.minute)

            descent_angle = np.arctan(self.descent_rate / true_air_speed).to(ureg.radian)
            horizontal_speed = (true_air_speed * np.cos(descent_angle)).to(ureg.nautical_mile / ureg.hour)
            horizontal_distance = (horizontal_speed * time_to_descend).to(ureg.nautical_mile)

            return time_to_descend, horizontal_distance

        except Exception as e:
            logging.error(f"Error in time_to_descend: {e}")
            raise
