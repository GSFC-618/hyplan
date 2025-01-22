import geopandas as gpd
import pandas as pd
import logging
from shapely.geometry import Point
from shapely.ops import nearest_points
from typing import List, Union
from hyplan.units import convert_distance
from hyplan.download import download_file
from hyplan.geometry import haversine

__all__ = [
    "Airport", "initialize_data", "find_nearest_airport", "airports_within_radius",
    "get_airports", "get_airport_details", "generate_geojson", "get_runway_details"
]

# Configurable file paths
AIRPORTS_FILE = "airports.csv"
RUNWAYS_FILE = "runways.csv"
OUR_AIRPORTS_URL = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv"
RUNWAYS_URL = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/runways.csv"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for airport and runway data
gdf_airports: gpd.GeoDataFrame = None
df_runways: pd.DataFrame = None

class Airport:
    """Class to represent an airport with Shapely Point geometry."""
    def __init__(self, icao: str):
        global gdf_airports
        if gdf_airports is None:
            raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")
        if icao not in gdf_airports.index:
            raise ValueError(f"Airport ICAO code {icao} not found in the dataset.")

        # Fetch airport data
        airport_data = gdf_airports.loc[icao]

        # Validate and extract longitude and latitude
        longitude = airport_data.get("longitude")
        latitude = airport_data.get("latitude")
        if pd.isna(longitude) or pd.isna(latitude):
            raise ValueError(f"Longitude or latitude is missing for airport {icao}")
        try:
            self.geometry = Point(float(longitude), float(latitude))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid longitude/latitude for airport {icao}: {longitude}, {latitude}")

        # Assign additional attributes
        self.icao = airport_data['icao_code']
        self.iata = airport_data['iata_code']
        self.name = airport_data['name']
        self.iso_country = airport_data['iso_country']
        self.municipality = airport_data['municipality']

    def __repr__(self):
        return f"<Airport {self.icao} - {self.name}>"

    @property
    def longitude(self):
        """Longitude of the airport."""
        return self.geometry.x

    @property
    def latitude(self):
        """Latitude of the airport."""
        return self.geometry.y


def generate_geojson(filepath: str = "airports.geojson", icao_codes: Union[str, List[str]] = None) -> None:
    """
    Generate a GeoJSON file of the airports using GeoPandas with CRS explicitly set to EPSG:4326.

    Args:
        filepath (str): Path to save the GeoJSON file. Defaults to "airports.geojson".
        icao_codes (Union[str, List[str]]): List of ICAO codes to subset the GeoJSON. If None, export all airports.
    """
    global gdf_airports
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")

    # Subset by ICAO codes if provided
    if icao_codes:
        if isinstance(icao_codes, str):
            icao_codes = [icao_codes]
        subset = gdf_airports[gdf_airports['icao_code'].isin(icao_codes)].copy()
    else:
        subset = gdf_airports.copy()

    # Ensure CRS is set to EPSG:4326
    if subset.crs is None:
        logging.warning("GeoDataFrame has no CRS. Setting CRS to EPSG:4326.")
        subset.set_crs("EPSG:4326", inplace=True)
    elif subset.crs.to_string() != "EPSG:4326":
        subset = subset.to_crs("EPSG:4326")

    # Write to GeoJSON using GeoPandas
    subset.to_file(filepath, driver="GeoJSON")
    logging.info(f"GeoJSON file generated at {filepath} with CRS EPSG:4326 and {len(subset)} airports.")


def _filter_airports_by_country(df_airports: pd.DataFrame, countries: List[str]) -> pd.DataFrame:
    """Filter airports by country codes."""
    return df_airports[df_airports['iso_country'].isin(countries)]

def _filter_airports_by_type(df_airports: pd.DataFrame, airport_types: List[str]) -> pd.DataFrame:
    """Filter airports by type."""
    return df_airports[df_airports['type'].isin(airport_types)]

def _filter_runways(df_runways: pd.DataFrame, length_ft: int = None, surface: Union[str, List[str]] = None, partial_match: bool = False) -> pd.DataFrame:
    """Filter runways based on length and/or surface type with optional partial matching."""
    if length_ft:
        df_runways = df_runways[df_runways['length_ft'] >= length_ft]
    if surface:
        if isinstance(surface, str):
            surface = [surface]
        if partial_match:
            pattern = "|".join([s.upper() for s in surface])
            df_runways = df_runways[df_runways['surface'].str.upper().str.contains(pattern, na=False)]
        else:
            df_runways = df_runways[df_runways['surface'].str.upper().isin([s.upper() for s in surface])]
    return df_runways

def load_airports(
    filepath: str = AIRPORTS_FILE,
    countries: List[str] = None,
    min_runway_length: int = None,
    runway_surface: Union[str, List[str]] = None,
    airport_types: List[str] = None
) -> gpd.GeoDataFrame:
    """Load and preprocess airport data with filters for country codes, runway length, surface type, and airport types."""
    try:
        df_airports = pd.read_csv(filepath, encoding="ISO-8859-1")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist. Please run download_file().")

    df_airports['icao_code'] = df_airports['ident']
    df_airports.rename(columns={"latitude_deg": "latitude", "longitude_deg": "longitude"}, inplace=True)
    df_airports.set_index('ident', inplace=True)

    columns_to_drop = ['id', 'elevation_ft', 'scheduled_service', 'local_code', 'gps_code',
                       'home_link', 'wikipedia_link', 'keywords']
    df_airports.drop(columns_to_drop, axis=1, inplace=True)

    # Filter by airport type
    if not airport_types:
        airport_types = ['large_airport', 'medium_airport', 'small_airport']
    df_airports = _filter_airports_by_type(df_airports, airport_types)

    # Filter by country codes
    if countries:
        df_airports = _filter_airports_by_country(df_airports, countries)

    # Filter by runway length and surface
    if min_runway_length or runway_surface:
        df_runways = load_runways()
        df_runways = _filter_runways(df_runways, length_ft=min_runway_length, surface=runway_surface)
        valid_airports = df_runways['airport_ident'].unique()
        df_airports = df_airports[df_airports['icao_code'].isin(valid_airports)]

    gdf_airports = gpd.GeoDataFrame(
        df_airports, geometry=gpd.points_from_xy(df_airports.longitude, df_airports.latitude))
    gdf_airports.sindex  # Create spatial index
    return gdf_airports

def load_runways(filepath: str = RUNWAYS_FILE) -> pd.DataFrame:
    """Load and preprocess runway data."""
    try:
        df_runways = pd.read_csv(filepath, encoding="ISO-8859-1")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist. Please run download_file().")

    columns_to_keep = ['airport_ident', 'length_ft', 'width_ft', 'surface']
    df_runways = df_runways[columns_to_keep]
    return df_runways

def initialize_data(countries: List[str] = None, min_runway_length: int = None, runway_surface: Union[str, List[str]] = None, airport_types: List[str] = None) -> None:
    """Initialize global variables for airports and runways data with filtering options."""
    global gdf_airports, df_runways
    download_file(AIRPORTS_FILE, OUR_AIRPORTS_URL, replace=False)
    download_file(RUNWAYS_FILE, RUNWAYS_URL, replace=False)
    gdf_airports = load_airports(
        countries=countries,
        min_runway_length=min_runway_length,
        runway_surface=runway_surface,
        airport_types=airport_types
    )
    df_runways = load_runways()

def find_nearest_airport(lat: float, lon: float, unit: str = "kilometers") -> str:
    """Find the nearest airport to a given latitude and longitude.

    Returns:
        str: ICAO code of the nearest airport.
    """
    global gdf_airports
    point = Point(lon, lat)
    nearest_geom = nearest_points(point, gdf_airports.unary_union)[1]
    nearest_airport = gdf_airports.loc[gdf_airports.geometry == nearest_geom].iloc[0]
    return nearest_airport['icao_code']

def airports_within_radius(lat: float, lon: float, radius: float, unit: str = "kilometers") -> List[str]:
    """Find all airports within a specified radius of a given point.

    Returns:
        List[str]: List of ICAO codes of airports within the radius.
    """
    global gdf_airports
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")

    point = Point(lon, lat)
    radius_m = convert_distance(radius, unit, "meters")

    buffer = point.buffer(radius_m / 111139.0)  # Approximate degree buffer
    possible_matches = gdf_airports[gdf_airports.intersects(buffer)].copy()

    possible_matches.loc[:, 'haversine_distance'] = possible_matches.apply(
        lambda row: haversine(lat, lon, row.latitude, row.longitude), axis=1
    )
    within_radius = possible_matches[possible_matches['haversine_distance'] <= radius_m]

    return within_radius['icao_code'].tolist()

def get_airports() -> gpd.GeoDataFrame:
    """Get the globally initialized GeoDataFrame of airports."""
    global gdf_airports
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")
    return gdf_airports

def get_airport_details(icao_codes: Union[str, List[str]]) -> pd.DataFrame:
    """Get details of airports for given ICAO code(s)."""
    global gdf_airports
    if isinstance(icao_codes, str):
        icao_codes = [icao_codes]
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")
    return gdf_airports[gdf_airports['icao_code'].isin(icao_codes)]

def generate_geojson(filepath: str = "airports.geojson", icao_codes: Union[str, List[str]] = None) -> None:
    """
    Generate a GeoJSON file of the airports using GeoPandas with CRS explicitly set to EPSG:4326.

    Args:
        filepath (str): Path to save the GeoJSON file. Defaults to "airports.geojson".
        icao_codes (Union[str, List[str]]): List of ICAO codes to subset the GeoJSON. If None, export all airports.
    """
    global gdf_airports
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")

    # Subset by ICAO codes if provided
    if icao_codes:
        if isinstance(icao_codes, str):
            icao_codes = [icao_codes]
        subset = gdf_airports[gdf_airports['icao_code'].isin(icao_codes)].copy()
    else:
        subset = gdf_airports.copy()

    # Ensure CRS is set to EPSG:4326
    if subset.crs is None:
        logging.warning("GeoDataFrame has no CRS. Setting CRS to EPSG:4326.")
        subset.set_crs("EPSG:4326", inplace=True)
    elif subset.crs.to_string() != "EPSG:4326":
        subset = subset.to_crs("EPSG:4326")

    # Write to GeoJSON using GeoPandas
    subset.to_file(filepath, driver="GeoJSON")
    logging.info(f"GeoJSON file generated at {filepath} with CRS EPSG:4326 and {len(subset)} airports.")

def get_runway_details(icao_codes: Union[str, List[str]]) -> pd.DataFrame:
    """
    Retrieve details of all runways for one or more airports.

    Args:
        icao_codes (Union[str, List[str]]): A single ICAO code or a list of ICAO codes.

    Returns:
        pd.DataFrame: A DataFrame with runway details for the given airport(s).
    """
    global df_runways
    if df_runways is None:
        raise RuntimeError("Runways data has not been initialized. Please run initialize_data().")

    if isinstance(icao_codes, str):
        icao_codes = [icao_codes]

    return df_runways[df_runways['airport_ident'].isin(icao_codes)]
