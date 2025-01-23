#%%

from hyplan.sun import solar_threshold_times

# Example usage:
df = solar_threshold_times(34.05, -118.25, '2025-03-01', '2025-05-07', [50], timezone_offset=-8)
print(df.to_markdown())

# %%
