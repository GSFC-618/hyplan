#%%
from datetime import datetime
import matplotlib.pyplot as plt
from hyplan.sun import solar_threshold_times, solar_azimuth, solar_position_increments, plot_solar_positions

lat = 9.615     # Example: Coastal Ocean off of Jaco, Costa Rica
lon = -84.82
min_elev = 35
max_elev = 55

# Example usage:
df = solar_threshold_times(lat, lon, '2025-02-01', '2025-03-07', [min_elev, max_elev], timezone_offset=-6)
print(df.to_markdown())

# %%

  
dt_example = datetime(2025, 2, 9, 17, 0, 0)  # 5:00 PM UTC
az = solar_azimuth(lat, lon, dt_example)
print(f"Solar azimuth at {dt_example} UTC: {az:.2f}°")

#%%

# Example for solar azimuth increments on a given day.
date_example = "2025-02-09"
min_elev = 35   # Only report times when the solar elevation is above 45 degrees.
df_positions = solar_position_increments(lat, lon, date_example, min_elev, timezone_offset=-6)
print(df_positions.to_markdown())
# %%

plot_solar_positions(df_positions)

