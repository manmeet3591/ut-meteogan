#!/usr/bin/env python
# coding: utf-8

# In[2]:


# get_ipython().system('which python')


# In[3]:


import sys
import xarray as xr
aorc_path = '/scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/'
graphcast_path = '/scratch/08105/ms86336/graphcast/'


# In[4]:


year = '2018/'

ds_graphcast = xr.open_dataset(graphcast_path+year+'graphcast_2018_01_01.nc')
# ds_graphcast


# In[5]:


input_vars = ["t2m",  "msl", "u10m", "v10m", \
              "u1000", "u925", "u850", "u700", "u500", "u250", \
              "v1000", "v925", "v850", "v700", "v500", "v250", \
             "w1000", "w925", "w850", "w700", "w500", "w250", \
              "z1000", "z925", "z850", "z700", "z500", "z200", \
              "t1000", "t925", "t850", "t700", "t500", "t100", 
              "q1000", "q925", "q850", "q700", "q500", "q100", "tp06"]


# In[6]:


# ds_graphcast[input_vars]


# In[7]:


ds_aorc = xr.open_dataset(aorc_path+'noaa_aorc_usa_2018.nc')
# ds_aorc


# In[8]:


vars_target = ['APCP_surface', 'DLWRF_surface', 'DSWRF_surface', 'PRES_surface', \
              'SPFH_2maboveground', 'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
# ds_aorc[vars_target]


# In[5]:


# ds_aorc.APCP_surface.isel(time=10).plot(cmap='Blues', vmax=6)


# # Compute Means 

# In[9]:


# Load AORC for 2019 and 2020
import xarray as xr
ds_aorc_19 = xr.open_mfdataset(aorc_path + 'noaa_aorc_usa_2019*.nc')
ds_aorc_20 = xr.open_mfdataset(aorc_path + 'noaa_aorc_usa_2020*.nc')
ds_aorc = xr.concat([ds_aorc_19, ds_aorc_20], dim='time')


# In[10]:


# Initialize dictionary to hold resampled variables
resampled_vars = {}

# Variables for summing
sum_vars = ['APCP_surface']

# Variables for averaging
mean_vars = [v for v in vars_target if v not in sum_vars]

# Sum precipitation
for var in sum_vars:
    resampled_vars[var] = ds_aorc[var].resample(time='6H').sum()

# Mean for other variables
for var in mean_vars:
    resampled_vars[var] = ds_aorc[var].resample(time='6H').mean()

# Combine into a new dataset
ds_aorc_6h = xr.Dataset(resampled_vars)


import json
import dask
from dask.diagnostics import ProgressBar

min_vals = {}
max_vals = {}

print("Preparing Dask tasks for min and max computation...")

# Collect delayed min/max computations
min_tasks = {}
max_tasks = {}

for var in ds_aorc_6h.data_vars:
    print(f"Queuing computation for '{var}'...")
    min_tasks[var] = ds_aorc_6h[var].min()
    max_tasks[var] = ds_aorc_6h[var].max()

# Compute all in parallel with progress bar
with ProgressBar():
    min_results, max_results = dask.compute(list(min_tasks.values()), list(max_tasks.values()))

# Store as plain floats
for var, min_val, max_val in zip(min_tasks.keys(), min_results, max_results):
    min_vals[var] = float(min_val)
    max_vals[var] = float(max_val)

print("\nCompleted computing min and max values.")

# Save to disk
with open('min_vals_aorc.json', 'w') as f_min:
    json.dump(min_vals, f_min)

with open('max_vals_aorc.json', 'w') as f_max:
    json.dump(max_vals, f_max)

print("Saved min_vals_aorc.json and max_vals_aorc.json.")



sys.exit()

# In[11]:


# ds_aorc_6h


# In[12]:


# ds_aorc_6h.isel(time=0).max().compute()


# # In[12]:


# print(ds_aorc_6h.APCP_surface.chunks)


# In[17]:


# da = ds_aorc_6h.APCP_surface.chunk({'time': -1, 'latitude': 1000, 'longitude': 1000})


# In[21]:


from dask.distributed import Client

try:
    client = Client()
except Exception:
    import dask
    dask.config.set(scheduler='threads')
    print("Falling back to threads scheduler.")


# In[22]:


import dask
dask.config.set(scheduler='threads')


# In[23]:


da = ds_aorc_6h.APCP_surface.chunk({'time': 100, 'latitude': 1000, 'longitude': 1000})
max_val = da.max().compute()
print("Max precipitation value:", max_val)


# In[14]:


input_vars = ["t2m",  "msl", "u10m", "v10m", \
              "u1000", "u925", "u850", "u700", "u500", "u250", \
              "v1000", "v925", "v850", "v700", "v500", "v250", \
             "w1000", "w925", "w850", "w700", "w500", "w250", \
              "z1000", "z925", "z850", "z700", "z500", "z200", \
              "t1000", "t925", "t850", "t700", "t500", "t100", 
              "q1000", "q925", "q850", "q700", "q500", "q100", "tp06"]
ds_graphcast = xr.open_dataset('/scratch/08105/ms86336/graphcast/2019/graphcast_2019_01_01.nc')[input_vars]
t2m_min = ds_graphcast.t2m.values.min()
t2m_max = ds_graphcast.t2m.values.max()


# In[17]:


import xarray as xr
import numpy as np
import glob
from tqdm import tqdm

input_vars = ["t2m",  "msl", "u10m", "v10m",
              "u1000", "u925", "u850", "u700", "u500", "u250",
              "v1000", "v925", "v850", "v700", "v500", "v250",
              "w1000", "w925", "w850", "w700", "w500", "w250",
              "z1000", "z925", "z850", "z700", "z500", "z200",
              "t1000", "t925", "t850", "t700", "t500", "t100",
              "q1000", "q925", "q850", "q700", "q500", "q100", "tp06"]

graphcast_files = sorted(glob.glob(graphcast_path + '2019/*.nc') + glob.glob(graphcast_path + '2020/*.nc'))

# Init dicts
min_dict = {var: np.inf for var in input_vars}
max_dict = {var: -np.inf for var in input_vars}

# Loop through files
for f in tqdm(graphcast_files, desc="Scanning GraphCast files"):
    try:
        for var in input_vars:
            ds = xr.open_dataset(f, engine='netcdf4')
            if var not in ds:
                continue
            data = ds[var].values
            if not np.isnan(data).all():
                min_dict[var] = min(min_dict[var], np.nanmin(data))
                max_dict[var] = max(max_dict[var], np.nanmax(data))
            ds.close()
    except Exception as e:
        print(f"⚠️ Skipped {f} due to error: {e}")


# In[21]:


import json
# Convert all values to float
min_dict_float = {k: float(v) for k, v in min_dict.items()}
max_dict_float = {k: float(v) for k, v in max_dict.items()}

# Save to JSON
with open("min_graphcast.json", "w") as f:
    json.dump(min_dict_float, f)
    
with open("max_graphcast.json", "w") as f:
    json.dump(max_dict_float, f)


# In[ ]:




