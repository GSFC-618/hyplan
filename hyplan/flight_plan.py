import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from .units import ureg
from .aircraft import Aircraft
from .airports import Airport
from .dubins_path import Waypoint, DubinsPath
from .flight_line import FlightLine

def compute_flight_plan(aircraft: Aircraft, flight_sequence: list, takeoff_airport: Airport, return_airport: Airport) -> gpd.GeoDataFrame:
    """
    Compute a flight plan given an aircraft, a sequence of FlightLines and/or WayPoints,
    and the departure/return airports.
    
    Returns a GeoDataFrame with flight segments including takeoff, cruise, and return phases.
    """
    records = []
    
    # Takeoff phase
    first_target = flight_sequence[0]
    if isinstance(first_target, FlightLine):
        first_target = first_target.waypoint1
    takeoff_info = aircraft.time_to_takeoff(takeoff_airport, first_target)
    records.extend(process_flight_phase(takeoff_airport, first_target, takeoff_info, "takeoff", "Departure"))
    
    # Iterate through the flight sequence
    i = 0
    while i < len(flight_sequence):
        segment = flight_sequence[i]
        
        if isinstance(segment, FlightLine):
            records.append(create_flight_line_record(segment, aircraft))
            if i + 1 < len(flight_sequence):
                next_segment = flight_sequence[i + 1]
                if isinstance(next_segment, FlightLine):
                    # Ensure smooth transition between flight lines
                    start = segment.waypoint2
                    end = next_segment.waypoint1
                    cruise_info = aircraft.time_to_cruise(start, end)
                    records.extend(process_flight_phase(start, end, cruise_info, "cruise", f"{start.name} to {end.name}"))
                else:
                    start = segment.waypoint2
                    cruise_info = aircraft.time_to_cruise(start, next_segment)
                    records.extend(process_flight_phase(start, next_segment, cruise_info, "cruise", f"{start.name} to {next_segment.name}"))
            i += 1
        else:
            if i + 1 < len(flight_sequence):
                start = segment
                end = flight_sequence[i + 1]
                if isinstance(end, FlightLine):
                    end = end.waypoint1
                cruise_info = aircraft.time_to_cruise(start, end)
                records.extend(process_flight_phase(start, end, cruise_info, "cruise", f"{start.name} to {end.name}"))
            i += 1
    
    # Return phase
    last_target = flight_sequence[-1]
    if isinstance(last_target, FlightLine):
        last_target = last_target.waypoint2
    return_info = aircraft.time_to_return(last_target, return_airport)
    records.extend(process_flight_phase(last_target, return_airport, return_info, "return", "Arrival"))
    
    # Create and return GeoDataFrame
    df = pd.DataFrame(records)
    flight_plan_gdf = gpd.GeoDataFrame(df, geometry=df["geometry"], crs="EPSG:4326")
    
    return flight_plan_gdf

def create_flight_line_record(flight_line, aircraft):
    """
    Helper function to create a flight line record for the DataFrame.
    """
    return {
        "geometry": flight_line.geometry,
        "start_lat": flight_line.lat1,
        "start_lon": flight_line.lon1,
        "end_lat": flight_line.lat2,
        "end_lon": flight_line.lon2,
        "start_altitude": flight_line.altitude.magnitude,
        "end_altitude": flight_line.altitude.magnitude,
        "start_heading": flight_line.waypoint1.heading,
        "end_heading": flight_line.waypoint2.heading,
        "time_to_segment": (flight_line.length / aircraft.cruise_speed).to(ureg.minute).magnitude,
        "segment_type": "flight_line",
        "segment_name": flight_line.site_name,
        "distance": flight_line.length.to(ureg.nautical_mile).magnitude
    }

def process_flight_phase(start, end, phase_info, segment_type, segment_name):
    """
    Process a flight phase and return structured records for the DataFrame.
    """
    records = []
    total_time = phase_info["total_time"].to(ureg.minute).magnitude
    dubins_path = phase_info["dubins_path"]
    
    for phase, details in phase_info["phases"].items():
        records.append({
            "geometry": dubins_path.geometry,
            "start_lat": start.latitude,
            "start_lon": start.longitude,
            "end_lat": end.latitude,
            "end_lon": end.longitude,
            "start_altitude": details["start_altitude"].to(ureg.meter).magnitude,
            "end_altitude": details["end_altitude"].to(ureg.meter).magnitude,
            "start_heading": getattr(start, "heading", None),
            "end_heading": getattr(end, "heading", None),
            "time_to_segment": (details["end_time"] - details["start_time"]).to(ureg.minute).magnitude,
            "segment_type": segment_type,
            "segment_name": segment_name,
            "distance": dubins_path.length.to(ureg.nautical_mile).magnitude
        })
    
    return records



def plot_flight_plan(flight_plan_gdf, takeoff_airport, return_airport, flight_sequence):
    """
    Plot the computed flight plan along with airports, waypoints, and flight lines.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    flight_plan_gdf.plot(ax=ax, column="segment_type", legend=True, cmap="viridis")
    
    # Plot takeoff and return airports
    ax.scatter(takeoff_airport.longitude, takeoff_airport.latitude, color='red', marker='*', s=200, label='Takeoff Airport')
    ax.scatter(return_airport.longitude, return_airport.latitude, color='blue', marker='*', s=200, label='Return Airport')
    
    # Plot waypoints
    for item in flight_sequence:
        if isinstance(item, Waypoint):
            ax.scatter(item.longitude, item.latitude, color='green', marker='o', s=100, label=item.name)
    
    # Plot flight lines
    for item in flight_sequence:
        if isinstance(item, FlightLine):
            x, y = zip(*item.geometry.coords)
            ax.plot(x, y, color='black', linestyle='dashed', linewidth=2, label=item.site_name)
    
    ax.set_title("Flight Plan")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend()
    plt.grid()
    plt.show()

def plot_altitude_trajectory(flight_plan_gdf):
    """
    Plot altitude vs. time trajectory.
    """
    plt.figure(figsize=(10, 5))
    
    cumulative_time = 0
    for _, row in flight_plan_gdf.iterrows():
        plt.plot(
            [cumulative_time, cumulative_time + row["time_to_segment"]],
            [row["start_altitude"], row["end_altitude"]],
            marker="o",
            label=row["segment_name"]
        )
        cumulative_time += row["time_to_segment"]
    
    plt.xlabel("Time (minutes)")
    plt.ylabel("Altitude (meters)")
    plt.title("Altitude vs. Time Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()
