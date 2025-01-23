#%%

from hyplan.sun import solar_threshold_times

# Example usage:
df = solar_threshold_times(9.62, -84.82, '2025-02-01', '2025-03-07', [35, 50], timezone_offset=-6)
print(df.to_markdown())

# %%
