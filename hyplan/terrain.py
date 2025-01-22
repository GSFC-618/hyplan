import logging
import os
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from shapely.geometry import box
from rtree import index
from osgeo import gdal
from typing import List, Tuple
import pymap3d.los
import pymap3d.aer

from .download import download_file

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_cache_root(custom_path: str = None) -> str:
    """
    Get the root directory for caching files.
    """
    return custom_path or os.environ.get("HYPLAN_CACHE_ROOT", f"{tempfile.gettempdir()}/hyplan")

def clear_cache():
    """
    Clears the entire cache directory after confirming it is safe to do so.
    """
    cache_dir = get_cache_root()
    if not cache_dir.startswith(tempfile.gettempdir()):
        raise ValueError(f"Refusing to clear unsafe cache directory: {cache_dir}")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logging.info(f"Cache directory {cache_dir} cleared.")
    else:
        logging.info(f"Cache directory {cache_dir} does not exist.")

def clear_localdem_cache(confirm: bool = True):
    """
    Clears the local DEM cache directory.

    This removes all files in the 'localdem' subdirectory of the cache root,
    which stores downloaded DEM tiles.

    Args:
        confirm (bool): If True, prompt the user for confirmation before clearing the cache.
    """
    localdem_dir = os.path.join(get_cache_root(), "localdem")
    
    # Check if the directory exists
    if not os.path.exists(localdem_dir):
        logging.info(f"Local DEM cache directory {localdem_dir} does not exist.")
        return
    
    # Safety check: Ensure the path starts within the cache root
    cache_root = get_cache_root()
    if not os.path.commonpath([localdem_dir, cache_root]) == cache_root:
        raise ValueError(f"Refusing to clear unsafe directory: {localdem_dir}")
    
    # Confirmation prompt
    if confirm:
        user_input = input(f"Are you sure you want to delete all files in {localdem_dir}? (yes/no): ").strip().lower()
        if user_input not in ("yes", "y"):
            logging.info("Local DEM cache clear operation canceled by the user.")
            return
    
    # Clear the directory
    try:
        shutil.rmtree(localdem_dir)
        logging.info(f"Local DEM cache directory {localdem_dir} cleared successfully.")
    except Exception as e:
        logging.error(f"Failed to clear local DEM cache: {e}")
        raise


def build_tile_index(tile_list_file: str) -> Tuple[index.Index, List[Tuple[str, box]]]:
    """
    Build an R-tree spatial index for DEM tiles.
    """
    idx = index.Index()
    tile_bboxes = []

    with open(tile_list_file) as file:
        for i, line in enumerate(file):
            tile = line.strip()
            try:
                lat, _, lon = tile.replace("_COG", "").split("_")[3:6]
                lon = -1 * float(lon[1:]) if "W" in lon else float(lon[1:])
                lat = -1 * float(lat[1:]) if "S" in lat else float(lat[1:])
                bbox = box(lon, lat, lon + 1, lat + 1)
                idx.insert(i, bbox.bounds)
                tile_bboxes.append((tile, bbox))
            except Exception as e:
                logging.warning(f"Skipping invalid tile entry: {tile} ({e})")

    return idx, tile_bboxes

def download_dem_files(lon_min: float, lat_min: float, lon_max: float, lat_max: float, aws_dir: str) -> List[str]:
    localdem_dir = os.path.join(get_cache_root(), "localdem")
    os.makedirs(localdem_dir, exist_ok=True)

    tile_list_file = os.path.join(localdem_dir, "tileList.txt")
    if not os.path.exists(tile_list_file):
        download_file(tile_list_file, f"{aws_dir}tileList.txt")

    idx, tile_bboxes = build_tile_index(tile_list_file)
    query_bbox = box(lon_min, lat_min, lon_max, lat_max)
    matching_tiles = [tile_bboxes[i][0] for i in idx.intersection(query_bbox.bounds)]

    if not matching_tiles:
        logging.info("No overlapping DEM tiles found.")
        return []

    downloaded_files = []
    with ThreadPoolExecutor() as executor:
        futures = {}
        for tile in matching_tiles:
            tile_url = f"{aws_dir}{tile}/{tile}.tif"
            tile_file = os.path.join(localdem_dir, f"{tile}.tif")
            if not os.path.exists(tile_file):
                logging.info(f"Submitting download for tile: {tile_url}")
                futures[executor.submit(download_file, tile_file, tile_url)] = tile_file
            else:
                downloaded_files.append(tile_file)

        for future, tile_file in futures.items():
            try:
                future.result()
                downloaded_files.append(tile_file)
            except Exception as e:
                logging.error(f"Error downloading tile {tile_file}: {e}")

    return downloaded_files


def merge_tiles(output_filename, tile_file_list):
    if not tile_file_list:
        raise ValueError("No tiles provided for merging.")

    invalid_tiles = [tile for tile in tile_file_list if not tile or not os.path.exists(tile)]
    if invalid_tiles:
        raise ValueError(f"Invalid or missing raster files: {invalid_tiles}")

    try:
        logging.info(f"Merging {len(tile_file_list)} tiles into {output_filename}")
        gdal.Warp(
            destNameOrDestDS=output_filename,
            srcDSOrSrcDSTab=tile_file_list,
            format="GTiff",
        )
        logging.info(f"Successfully merged tiles into {output_filename}")
    except Exception as e:
        logging.error(f"Failed to merge tiles: {e}")
        raise RuntimeError(f"Tile merging failed: {e}")



def generate_demfile(latitude: np.ndarray, longitude: np.ndarray, aws_dir: str = "https://copernicus-dem-30m.s3.amazonaws.com/") -> str:
    """
    Generate a DEM file covering the specified latitude and longitude extents.
    """
    dem_cache_dir = os.path.join(get_cache_root(), "dem_cache")
    os.makedirs(dem_cache_dir, exist_ok=True)

    lon_min, lon_max = np.min(longitude) - 0.1, np.max(longitude) + 0.1
    lat_min, lat_max = np.min(latitude) - 0.1, np.max(latitude) + 0.1

    cache_filename = os.path.join(dem_cache_dir, f"{int(lat_min)}_{int(lon_min)}_{int(lat_max)}_{int(lon_max)}.tif")
    if os.path.exists(cache_filename):
        logging.info(f"Using cached DEM file: {cache_filename}")
        return cache_filename

    tile_files = download_dem_files(lon_min, lat_min, lon_max, lat_max, aws_dir)
    if not tile_files:
        raise FileNotFoundError("No DEM tiles available for the specified area.")

    merge_tiles(cache_filename, tile_files)
    return cache_filename

def get_elevations(lats: np.ndarray, lons: np.ndarray, dem_file: str) -> np.ndarray:
    """
    Extract elevation values for given latitudes and longitudes from a DEM file.
    """
    dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
    if not dataset:
        raise RuntimeError(f"Could not open DEM file: {dem_file}")

    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    if not band:
        raise RuntimeError(f"DEM file does not contain valid raster data: {dem_file}")

    def get_pixel(lat, lon):
        x = int((lon - geotransform[0]) / geotransform[1])
        y = int((lat - geotransform[3]) / geotransform[5])
        return x, y

    elevations = []
    for lat, lon in zip(lats, lons):
        x, y = get_pixel(lat, lon)
        elevations.append(band.ReadAsArray(x, y, 1, 1)[0][0])

    return np.array(elevations)


def get_min_max_elevations(dem_file: str) -> float:
    """
    Get the maximum elevation value from a DEM file.

    Args:
        dem_file (str): Path to the DEM file.

    Returns:
        float: Maximum elevation value in the DEM file.
    """
    # Open the DEM file
    dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
    if not dataset:
        raise RuntimeError(f"Could not open DEM file: {dem_file}")

    # Read the raster band
    band = dataset.GetRasterBand(1)
    if not band:
        raise RuntimeError(f"DEM file does not contain valid raster data: {dem_file}")

    # Get the minimum and maximum elevation values
    min_val, max_val = band.ComputeRasterMinMax()
    dataset = None  # Close the dataset

    return min_val, max_val

def ray_terrain_intersection(
    lat0: np.ndarray,
    lon0: np.ndarray,
    h0: float,
    az: float,
    tilt: float,
    precision: float = 10.0,
    dem_file: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch computation of ray-terrain intersections using a DEM for multiple observer positions.
    Vectorized to handle multiple observers efficiently.

    Args:
        lat0 (np.ndarray): Array of observer latitudes (degrees).
        lon0 (np.ndarray): Array of observer longitudes (degrees).
        h0 (float): Altitude of the observer above the ellipsoid (meters).
        az (float): Azimuth angle of the ray (degrees).
        tilt (float): Tilt angle of the ray (degrees).
        precision (float): Precision of the slant range sampling (meters).
        dem_file (str): Path to the DEM file. If None, it will generate one.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (intersection_lats, intersection_lons, intersection_alts)
    """
    # Validate input
    if np.any((tilt < -90) | (tilt > 90)):
        raise ValueError("Tilt angles must be between -90 and 90 degrees.")
    if not (0 <= az <= 360):
        raise ValueError("Azimuth angle must be between 0 and 360 degrees.")

    # Compute slant range for ellipsoid intersection
    lat_ell, lon_ell, rng_ell = pymap3d.los.lookAtSpheroid(lat0, lon0, h0, az, tilt)

    # Generate DEM file if not provided
    if dem_file is None:
        dem_file = generate_demfile(lat_ell, lon_ell)

    # Get terrain elevation bounds
    min_elev, max_elev = get_min_max_elevations(dem_file)
    max_elev = min(h0, max_elev)  # Cap maximum elevation at observer altitude
    if np.any(min_elev > h0):
        raise ValueError("Observer altitude is below the minimum terrain elevation.")

    # Compute slant range bounds
    cos_tilt = np.cos(np.radians(tilt))
    upper_bound = rng_ell - (min_elev / cos_tilt)
    lower_bound = rng_ell - (max_elev / cos_tilt)

    lower_bound = np.floor(lower_bound / precision) * precision
    upper_bound = np.ceil(upper_bound / precision) * precision

    # Generate slant range sampling
    rs = np.arange(lower_bound.min(), upper_bound.max() + precision, precision)

    # Compute geodetic positions for all observer positions and slant ranges
    lats, lons, alts = pymap3d.aer.aer2geodetic(
        az, tilt - 90.0, rs[:, np.newaxis], lat0[np.newaxis, :], lon0[np.newaxis, :], h0
    )

    # Flatten for DEM query
    lats_flat = lats.ravel()
    lons_flat = lons.ravel()

    # Query DEM for terrain elevations
    dem_elevations = get_elevations(lats_flat, lons_flat, dem_file).reshape(lats.shape)

    # Find the first intersection point
    mask = dem_elevations > alts
    idx = np.argmax(mask, axis=0)

    # Extract intersection points
    intersection_lats = lats[idx, np.arange(len(lat0))]
    intersection_lons = lons[idx, np.arange(len(lon0))]
    intersection_alts = dem_elevations[idx, np.arange(len(lat0))]

    return intersection_lats, intersection_lons, intersection_alts
