
#%%
"""
HyPlan Clouds

Overview:
Optical remote sensing of the Earth's surface often requires clear skies. Deploying airborne remote sensing 
instruments can be costly, with daily costs for aircraft, labor, and per diem travel expenses for aircraft 
and instrument teams. This script addresses the question: 
"Statistically, how many days is it likely to take to acquire clear-sky observations for a given set of flight boxes?".

The script operates under several simplifying assumptions:
- Each flight box is "flyable" in a single day given clear skies from a single base of operations.
- Instantaneous MODIS Terra/Aqua overpasses are representative of clear-sky conditions throughout the flight day.
- Other environmental state parameters (e.g., tides, wind speeds) do not influence go/no-go decisions for flights.

Key Features:
1. **Cloud Data Processing**:
    - Reads geospatial polygon data (GeoJSON) representing flight areas for airborne optical remote sensing.
    - Fetches MODIS cloud fraction data from Google Earth Engine for specified years and day ranges.
    - Aggregates daily cloud fraction data for each polygon.

2. **Flight Simulation**:
    - Simulates daily flight schedules to visit polygons based on a maximum cloud fraction threshold.
    - Enforces constraints such as maximum consecutive flight days and optional weekend exclusions.

3. **Visualization**:
    - Generates heatmaps of cloud conditions, visit days, and rest days for each simulated year.
    - Produces cumulative distribution function (CDF) plots to estimate the likelihood of completing visits.

TODO: Currently, it is not possible to span years (a campaign can not include dates in both December and January). 
"""
#%%

# Core Libraries
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict

# Geospatial Libraries
import geopandas as gpd
from shapely import wkb

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Google Earth Engine
import ee


# Initialize the Earth Engine with error handling
try:
    # ee.Authenticate()
    ee.Initialize()
except Exception as e:
    raise RuntimeError("Earth Engine initialization failed. Check your authentication.") from e

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)


#%%

def get_binary_cloud(image):
    """
    Generates a binary cloud mask for a given MODIS image.

    Parameters:
        image (ee.Image): An Earth Engine image with a "state_1km" quality assessment band.

    Returns:
        ee.Image: Binary cloud mask (1 for cloudy, 0 for clear) with an added "date_char" property.
    """
    qa = image.select("state_1km")
    clouds = qa.bitwiseAnd(1).eq(1)
    date_char = image.date().format('yyyy-MM-dd')
    return clouds.set("date_char", date_char)

def calculate_cloud_fraction(image, polygon_geometry):
    """
    Calculates the cloud fraction over a given polygon for a MODIS image.

    Parameters:
        image (ee.Image): An Earth Engine MODIS image.
        polygon_geometry (ee.Geometry): A polygon geometry representing the region of interest.

    Returns:
        ee.Feature: A feature containing the date and calculated cloud fraction for the polygon.
    """
    # Reduce the cloud image to the region of interest and calculate mean
    reduction = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon_geometry,
        scale=1000
    )
    # Extract the cloud fraction value
    cloud_fraction = reduction.get('state_1km')
    # Return the cloud fraction value with date
    return ee.Feature(None, {'date_char': image.get('date_char'), 'cloud_fraction': cloud_fraction})

# Determine year spans based on the day range
def create_date_ranges(day_start, day_stop, year_start, year_stop):
    """
    Creates date ranges for filtering Earth Engine image collections.

    Parameters:
        day_start (int): Start day of the year (1-365).
        day_stop (int): End day of the year (1-365).
        year_start (int): Start year for the ranges.
        year_stop (int): End year for the ranges.

    Returns:
        list of tuples: A list of date range tuples (start_date, end_date) in YYYY-DDD format.
    """
    date_ranges = []
    for year in range(year_start, year_stop + 1):
        # Adjust end date to ensure inclusion of day_stop
        end_date = f"{year}-{day_stop:03}"
        date_ranges.append((f"{year}-{day_start:03}", end_date))
    return date_ranges

def create_cloud_data_array_with_limit(polygon_file, year_start, year_stop, day_start, day_stop, limit=5000):
    """
    Processes MODIS cloud data for polygons and calculates daily cloud fractions.

    Parameters:
        polygon_file (str): Path to a GeoJSON or shapefile containing polygons.
        year_start (int): Start year for data processing.
        year_stop (int): End year for data processing.
        day_start (int): Start day of the year for data processing.
        day_stop (int): End day of the year for data processing.
        limit (int, optional): Maximum number of images to process per date range. Default is 5000.

    Returns:
        pd.DataFrame: A DataFrame with columns 'polygon_id', 'year', 'day_of_year', and 'cloud_fraction'.
    """
    try:
        # Load your polygon data
        gdf = gpd.read_file(polygon_file)
        if gdf.empty:
            raise ValueError("Polygon file is empty or invalid.")
    except Exception as e:
        raise RuntimeError(f"Failed to load polygon file: {polygon_file}") from e

    gdf = gdf[['Name', 'geometry']]
    _drop_z = lambda geom: wkb.loads(wkb.dumps(geom, output_dimension=2))
    gdf.geometry = gdf.geometry.apply(_drop_z)

    # Initialize results list
    results = []

    # Create date ranges based on the year spans
    date_ranges = create_date_ranges(day_start, day_stop, year_start, year_stop)

    # Initialize an empty ImageCollection
    cloud_data = ee.ImageCollection([])

    try:
        # Filter the collection by created date ranges and day of year
        for start, stop in date_ranges:
            cloud_aqua = ee.ImageCollection("MODIS/061/MOD09GA").filterDate(start, stop).limit(limit)
            cloud_terra = ee.ImageCollection('MODIS/061/MYD09GA').filterDate(start, stop).limit(limit)
            cloud_data = cloud_data.merge(cloud_aqua).merge(cloud_terra)

        # Map the binary cloud mask function
        cloud_data = cloud_data.map(get_binary_cloud)
    except Exception as e:
        raise RuntimeError("Error occurred while processing MODIS data.") from e

    # Iterate over each polygon
    for _, row in gdf.iterrows():
        polygon_name = row['Name']
        polygon_geojson = row['geometry'].__geo_interface__
        polygon_geometry = ee.Geometry(polygon_geojson)

        # Map the cloud fraction calculation over the image collection
        mapped_results = cloud_data.map(lambda image: calculate_cloud_fraction(image, polygon_geometry))

        # Get the results as a list of features
        feature_collection = ee.FeatureCollection(mapped_results)
        feature_list = feature_collection.limit(limit).getInfo()['features']

        # Process each feature
        for feature in feature_list:
            properties = feature['properties']
            date_char = properties.get('date_char')
            if date_char is None:
                continue  # Skip this feature if 'date_char' is not present
            cloud_fraction = properties.get('cloud_fraction')
            if cloud_fraction is not None:
                year, month, day = [int(x) for x in date_char.split('-')]
                day_of_year = pd.Timestamp(year=year, month=month, day=day).dayofyear
                results.append({'year': year, 'day_of_year': day_of_year, 'polygon_id': polygon_name, 'cloud_fraction': cloud_fraction})

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Aggregate the DataFrame to ensure unique indices
    # You can choose different aggregation methods like mean, median, etc.
    aggregated_df = results_df.groupby(['polygon_id', 'year', 'day_of_year']).mean().reset_index()
    return aggregated_df

def simulate_visits(
    df: pd.DataFrame,
    day_start: int,
    day_stop: int,
    year_start: int,
    year_stop: int,
    cloud_fraction_threshold: float = 0.10,
    rest_day_threshold: int = 6,
    exclude_weekends: bool = False,
    debug: bool = False
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, list]], Dict[int, list]]:
    """
    Simulate visits to polygons based on cloud fraction thresholds, ensuring no more than one visit per day.
    Adds rest days after a set number of consecutive visits and resets counters on weekends or when no polygons meet the threshold.

    Parameters:
        df (pd.DataFrame): Cloud fraction data with columns: 'polygon_id', 'year', 'day_of_year', 'cloud_fraction'.
        day_start (int): Start day of the year for simulation.
        day_stop (int): End day of the year for simulation.
        year_start (int): Start year for simulation.
        year_stop (int): End year for simulation.
        cloud_fraction_threshold (float): Maximum allowable cloud fraction for a visit.
        rest_day_threshold (int): Maximum number of consecutive visits before a rest day is required.
        exclude_weekends (bool): If True, skip weekends and reset the counter for rest days.
        debug (bool): If True, enable detailed logging for debugging.

    Returns:
        Tuple[pd.DataFrame, Dict[int, Dict[str, list]], Dict[int, list]]:
            - DataFrame summarizing total days simulated per year.
            - Dictionary of visit days for each polygon, organized by year.
            - Dictionary of rest days for each year.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    visit_days = []
    visit_tracker = {}
    rest_days = {}

    for year in range(year_start, year_stop + 1):
        visited_polygons = set()
        remaining_polygons = set(df['polygon_id'].unique())
        visit_tracker[year] = {}
        rest_days[year] = []
        current_day_of_year = day_start
        total_days = 0
        consecutive_visits = 0

        while current_day_of_year <= day_stop:
            total_days += 1
            current_date = datetime(year, 1, 1) + timedelta(days=current_day_of_year - 1)

            # Skip weekends if exclude_weekends=True
            if exclude_weekends and current_date.weekday() >= 5:
                logging.debug(f"Skipping weekend on day {current_day_of_year} of year {year}")
                current_day_of_year += 1
                consecutive_visits = 0  # Reset counter
                continue

            # Filter cloud data for polygons not yet visited
            daily_df = df[(df['year'] == year) & (df['day_of_year'] == current_day_of_year)]
            daily_df = daily_df[~daily_df['polygon_id'].isin(visited_polygons)]

            # Check if any polygons are visitable on this day
            visitable_polygons = daily_df[daily_df['cloud_fraction'] < cloud_fraction_threshold]

            if not visitable_polygons.empty:
                if consecutive_visits < rest_day_threshold:
                    # Visit the first polygon (sorted by priority, e.g., alphabetical order)
                    polygon_to_visit = visitable_polygons.sort_values(by='polygon_id').iloc[0]
                    polygon_id = polygon_to_visit['polygon_id']

                    visited_polygons.add(polygon_id)
                    remaining_polygons.discard(polygon_id)

                    if polygon_id not in visit_tracker[year]:
                        visit_tracker[year][polygon_id] = []
                    visit_tracker[year][polygon_id].append(current_day_of_year)

                    logging.debug(f"Visiting polygon {polygon_id} on day {current_day_of_year} of year {year}")

                    consecutive_visits += 1
                else:
                    # Trigger a rest day if threshold is reached
                    rest_days[year].append(current_day_of_year)
                    logging.info(f"Rest day added on day {current_day_of_year} of year {year}")
                    consecutive_visits = 0
            else:
                # No polygons meet the threshold; reset consecutive visits counter
                logging.debug(f"No visitable polygons on day {current_day_of_year} of year {year}")
                consecutive_visits = 0

            # Check if all polygons have been visited
            if not remaining_polygons:
                logging.info(f"All polygons visited for year {year}.")
                break

            current_day_of_year += 1

        # Append total days for the year
        visit_days.append({'year': year, 'days': total_days})

    return pd.DataFrame(visit_days), visit_tracker, rest_days

def plot_yearly_cloud_fraction_heatmaps_with_visits(
    cloud_data_df, visit_tracker, rest_days, 
    cloud_fraction_threshold=0.10, exclude_weekends=False, 
    day_start=1, day_stop=365
):
    """
    Generates heatmaps of cloud fraction for each year, including visit markers and rest day highlights.

    Parameters:
        cloud_data_df (pd.DataFrame): DataFrame with columns 'polygon_id', 'year', 'day_of_year', and 'cloud_fraction'.
        visit_tracker (dict): A dictionary of visit days for each polygon, organized by year.
        rest_days (dict): A dictionary of rest days for each year.
        cloud_fraction_threshold (float): Threshold to classify cloud fraction as clear (white) or cloudy (black).
        exclude_weekends (bool): If True, weekends are highlighted and skipped in the heatmap.
        day_start (int): Start day of the year to include in the heatmap.
        day_stop (int): End day of the year to include in the heatmap.

    Returns:
        None: Displays heatmaps for each year with clear/cloudy days, visit days, and rest day markers.
    """
    required_columns = {'polygon_id', 'year', 'day_of_year', 'cloud_fraction'}
    if not required_columns.issubset(cloud_data_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Define a custom colormap: white (clear), black (cloudy), grey (visited), purple (weekend), orange (rest days)
    cmap = mcolors.ListedColormap(['white', 'black', 'grey', 'purple', 'orange'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create a heatmap for each year
    unique_years = cloud_data_df['year'].unique()
    for year in sorted(unique_years):
        # Filter data for the year and restrict to the specified day range
        year_data = cloud_data_df[(cloud_data_df['year'] == year) & 
                                  (cloud_data_df['day_of_year'] >= day_start) & 
                                  (cloud_data_df['day_of_year'] <= day_stop)]
        heatmap_data = year_data.pivot(index='polygon_id', columns='day_of_year', values='cloud_fraction')

        # Ensure the heatmap only includes the specified range of days
        heatmap_data = heatmap_data.reindex(columns=range(day_start, day_stop + 1), fill_value=0)

        # Apply cloud fraction threshold to determine clear/cloudy
        binary_data = (heatmap_data >= cloud_fraction_threshold).astype(int)  # 1 = cloudy (black), 0 = clear (white)
        status_data = binary_data.copy()  # Start with clear/cloudy data

        # Process visits and mark visited days as grey
        stars_x = []
        stars_y = []
        rest_days_set = set(rest_days.get(year, [])) if rest_days else set()

        for i, polygon_id in enumerate(status_data.index):
            if polygon_id in visit_tracker.get(year, {}):
                visit_days = sorted(visit_tracker[year][polygon_id])
                for visit_day in visit_days:
                    if day_start <= visit_day <= day_stop:
                        stars_x.append(visit_day - day_start + 0.5)  # Adjust star position to match day_start
                        stars_y.append(i + 0.5)  # Center the star in the cell

                        # Mark all subsequent days as gray until another visit or the end of the range
                        for day in range(visit_day + 1, day_stop + 1):
                            if exclude_weekends:
                                weekday = (datetime(year, 1, 1) + timedelta(days=day - 1)).weekday()
                                if weekday < 5:  # Only mark weekdays
                                    status_data.loc[polygon_id, day] = 2  # Grey
                            else:
                                status_data.loc[polygon_id, day] = 2  # Grey

        # Mark rest days as orange
        for rest_day in rest_days_set:
            if day_start <= rest_day <= day_stop:
                status_data.iloc[:, rest_day - day_start] = 4  # Orange for rest days

        # Mark weekends as purple if exclude_weekends=True
        if exclude_weekends:
            for day in range(day_start, day_stop + 1):
                weekday = (datetime(year, 1, 1) + timedelta(days=day - 1)).weekday()
                if weekday >= 5:  # Saturday or Sunday
                    status_data.loc[:, day] = 3  # Purple for weekends

        # Plot the heatmap with the custom colormap
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(status_data, cmap=cmap, norm=norm, cbar=False, 
                         linewidths=0.5, linecolor='gray', square=True)
        plt.scatter(stars_x, stars_y, color='red', marker='*', s=150, label='Visit Day')
        plt.title(f'Cloud Fraction Heatmap with Visits for Year {year}')
        plt.xlabel('Day of Year')
        plt.ylabel('Polygon ID')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
