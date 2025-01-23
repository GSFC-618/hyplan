import pandas as pd
import numpy as np
from sunposition import sunpos
from datetime import datetime, timedelta

def solar_threshold_times(latitude, longitude, start_date, end_date, thresholds, timezone_offset=0):
    """
    Find times when the solar elevation crosses specified thresholds.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        thresholds (list): List of 1 or 2 solar elevation thresholds in degrees (e.g., [35] or [35, 50]).
        timezone_offset (int): Timezone offset from UTC in hours (e.g., -8 for PST, 1 for CET).

    Returns:
        pandas.DataFrame: DataFrame with reordered columns: ['Date', 'Rise_<lower>', 'Rise_<upper>', 'Set_<upper>', 'Set_<lower>'].
    """
    # Ensure thresholds is a list of 1 or 2 elements
    if not (1 <= len(thresholds) <= 2):
        raise ValueError("Thresholds must be a list with 1 or 2 elements.")

    # Generate all timestamps at 1-minute intervals in UTC
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    timestamps = pd.date_range(start=start_datetime, end=end_datetime + timedelta(days=1) - timedelta(minutes=1), freq='1min', tz='UTC')

    # Adjust timestamps to local timezone
    local_timestamps = timestamps + pd.Timedelta(hours=timezone_offset)

    # Vectorized calculation of solar positions using UTC timestamps
    _, zenith, *_ = sunpos(timestamps, latitude, longitude, elevation=0)
    elevation = 90 - zenith

    # Prepare DataFrame for results
    results = []

    # Iterate over each day
    for day in pd.date_range(start=start_datetime, end=end_datetime, freq='D'):
        day_mask = (local_timestamps.date == day.date())
        daily_times = local_timestamps[day_mask]
        daily_elevation = elevation[day_mask]

        if len(daily_times) == 0:
            continue  # Skip days with no data

        day_results = {'Date': day.date()}

        if len(thresholds) == 2:
            lower, upper = sorted(thresholds)
            for threshold, label in zip([lower, upper], [f'_{lower}', f'_{upper}']):
                rise_time = None
                fall_time = None

                above_threshold = daily_elevation > threshold

                if np.any(above_threshold):
                    rise_idx = np.argmax(above_threshold)  # First True
                    fall_idx = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1  # Last True

                    rise_time = daily_times[rise_idx].strftime('%H:%M:%S')
                    fall_time = daily_times[fall_idx].strftime('%H:%M:%S')

                day_results[f'Rise{label}'] = rise_time
                day_results[f'Set{label}'] = fall_time

        elif len(thresholds) == 1:
            lower = thresholds[0]
            day_results[f'Rise_{lower}'] = None
            day_results[f'Set_{lower}'] = None

            above_threshold = daily_elevation > lower

            if np.any(above_threshold):
                rise_idx = np.argmax(above_threshold)  # First True
                fall_idx = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1  # Last True

                day_results[f'Rise_{lower}'] = daily_times[rise_idx].strftime('%H:%M:%S')
                day_results[f'Set_{lower}'] = daily_times[fall_idx].strftime('%H:%M:%S')

        results.append(day_results)

    # Reorder and return DataFrame
    if len(thresholds) == 2:
        lower, upper = sorted(thresholds)
        columns = ['Date', f'Rise_{lower}', f'Rise_{upper}', f'Set_{upper}', f'Set_{lower}']
    else:
        columns = ['Date', f'Rise_{lower}', f'Set_{lower}']

    return pd.DataFrame(results, columns=columns)
