import json

import wandb
import matplotlib.pyplot as plt  # you already import earlier in your script sometimes
# ---- put these two lines at the very top of your script, before importing xarray/netCDF4 ----
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim

from tqdm import tqdm
import os, re, json, glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

# --- tiny helpers (add once near the top) ---
import numpy as np, xarray as xr, matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import torch
import torch.nn.functional as F

# --- open AORC for day 0 + next 2 days (hourly), robustly ---
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch
import torch.nn.functional as F

# # Create random input tensor with 1 channel
# input_tensor = torch.randn(1, 1, 64, 64)  # Batch size of 1, 1 channel, 64x64 dimensions

# Instantiate the Generator and Discriminator models
from srgan import SRResNet, Discriminator, compute_gradient_penalty, get_grads_G, get_grads_D_WAN


import torch
from tqdm import tqdm
from torch import nn
import torch.optim as optim

# ---- graphcast (your values) ----
graphcast_max = {"t2m": 325.4837341308594, "msl": 108430.8671875, "u10m": 34.456207275390625, "v10m": 37.61301040649414, "u1000": 35.95341873168945, "u925": 55.74516296386719, "u850": 57.068603515625, "u700": 62.47403335571289, "u500": 83.98092651367188, "u250": 119.2408218383789, "v1000": 36.69783020019531, "v925": 61.703094482421875, "v850": 61.97074890136719, "v700": 59.54432678222656, "v500": 80.41673278808594, "v250": 104.57147216796875, "w1000": 12.027145385742188, "w925": 12.556365966796875, "w850": 14.195926666259766, "w700": 18.170019149780273, "w500": 23.350290298461914, "w250": 10.038920402526855, "z1000": 5493.59814453125, "z925": 11000.4658203125, "z850": 17558.44140625, "z700": 32654.23046875, "z500": 59142.55859375, "z200": 124625.0859375, "t1000": 326.3714294433594, "t925": 321.5543518066406, "t850": 316.6293029785156, "t700": 301.09588623046875, "t500": 285.8623962402344, "t100": 243.3773651123047, "q1000": 0.030631616711616516, "q925": 0.025094827637076378, "q850": 0.02299940027296543, "q700": 0.017240697517991066, "q500": 0.009967059828341007, "q100": 4.436284507391974e-05, "tp06": 0.1836116909980774}
graphcast_min = {"t2m": 190.68714904785156, "msl": 91563.171875, "u10m": -41.4630126953125, "v10m": -35.92013931274414, "u1000": -39.79793167114258, "u925": -58.45619201660156, "u850": -66.82825469970703, "u700": -61.006805419921875, "u500": -55.76197052001953, "u250": -69.47474670410156, "v1000": -36.5177116394043, "v925": -55.0919189453125, "v850": -55.936492919921875, "v700": -53.18024444580078, "v500": -69.1786117553711, "v250": -98.36532592773438, "w1000": -6.234199047088623, "w925": -9.378058433532715, "w850": -14.728086471557617, "w700": -19.180068969726562, "w500": -23.075439453125, "w250": -16.901153564453125, "z1000": -6866.56494140625, "z925": -873.0474243164062, "z850": 5524.56396484375, "z700": 19814.2421875, "z500": 42841.69140625, "z200": 99689.03125, "t1000": 214.81982421875, "t925": 217.47918701171875, "t850": 214.00636291503906, "t700": 202.0097198486328, "t500": 213.86427307128906, "t100": 176.72293090820312, "q1000": -0.0025864115450531244, "q925": -0.0020713885314762592, "q850": -0.0017936615040525794, "q700": -0.0012956960126757622, "q500": -0.0008684174390509725, "q100": -0.00013288730406202376, "tp06": -0.0026905916165560484}

# ---- aorc (your values) ----
aorc_max = {"APCP_surface": 388.400005787611, "DLWRF_surface": 526.4000078439713, "DSWRF_surface": 1215.083351439486, "PRES_surface": 105375.0, "SPFH_2maboveground": 0.028899999269924592, "TMP_2maboveground": 326.25000486150384, "UGRD_10maboveground": 28.433333757023018, "VGRD_10maboveground": 34.41666717951497}
aorc_min = {"APCP_surface": 0.0, "DLWRF_surface": 90.13333467642467, "DSWRF_surface": 0.0, "PRES_surface": 58343.333333333336, "SPFH_2maboveground": 9.999999747378752e-05, "TMP_2maboveground": 227.03333671639362, "UGRD_10maboveground": -27.63333374510209, "VGRD_10maboveground": -29.666667108734448}


# --- evaluation config ---
DELTA_SSIM       = 1e-3                    # improvement margin
EVAL_DATE        = "2021-09-01"            # heavy rain (Ida NE)
EVAL_BOX         = {"lat_min": 35.0, "lat_max": 45.0, "lon_min": -80.0, "lon_max": -70.0}
EVAL_PATCH_SIZE  = 128
EVAL_BATCH       = 16                      # patch inference batch
EVAL_TIMES_TO_LOG = (0, 6, 12, 24, 48, 71) # fewer images to keep it light

# with open("graphcast_minmax_2018-2020.json", "w") as f:
#     json.dump({"dataset": "graphcast", "years": [2018, 2019, 2020], "min": graphcast_min, "max": graphcast_max}, f, indent=2)

# with open("aorc_minmax_2018-2020.json", "w") as f:
#     json.dump({"dataset": "aorc", "years": [2018, 2019, 2020], "min": aorc_min, "max": aorc_max}, f, indent=2)

# print("Wrote graphcast_minmax_2018-2020.json and aorc_minmax_2018-2020.json")
# #print(graphcast_max)



def fix_coords(ds):
    if "latitude" in ds.dims: ds = ds.rename({"latitude":"lat"})
    if "longitude" in ds.dims: ds = ds.rename({"longitude":"lon"})
    if "latitude" in ds.variables: ds = ds.rename({"latitude":"lat"})
    if "longitude" in ds.variables: ds = ds.rename({"longitude":"lon"})
    return ds

def sort_gc_lonlat(ds_gc):
    ds_gc = ds_gc.assign_coords(lon=((ds_gc["lon"] + 180) % 360) - 180).sortby("lon")
    if ds_gc["lat"][0] > ds_gc["lat"][-1]:
        ds_gc = ds_gc.sortby("lat")
    return ds_gc

def read_aorc_3days(base, d0):
    files, dsets, last_err = [], [], None
    for i in range(3):
        di = d0 + timedelta(days=i)
        ymd = di.strftime("%Y%m%d")
        files.append(f"{base}/noaa_aorc_usa_{di.year}_day_{ymd}.nc")
    for p in files:
        ok = False
        for eng in ("netcdf4", "h5netcdf"):
            try:
                ds_i = xr.open_dataset(p, engine=eng, chunks={"time": 24})
                _ = ds_i["time"].isel(time=0).values
                dsets.append(ds_i); ok = True; break
            except Exception as e:
                last_err = e
        if not ok: print(f"[skip] {Path(p).name}: {last_err}")
    if not dsets: raise RuntimeError("No AORC files for eval window")
    return fix_coords(xr.combine_by_coords(dsets, combine_attrs="override"))

def box_indices(lat_arr, lon_arr, box, patch):
    lat, lon = np.asarray(lat_arr), np.asarray(lon_arr)
    i0 = np.searchsorted(lat, box["lat_min"], "left")
    i1 = np.searchsorted(lat, box["lat_max"], "right") - 1
    j0 = np.searchsorted(lon, box["lon_min"], "left")
    j1 = np.searchsorted(lon, box["lon_max"], "right") - 1
    H = i1 - i0 + 1; W = j1 - j0 + 1
    Hc = H - (H % patch); Wc = W - (W % patch)
    if Hc <= 0 or Wc <= 0:
        raise ValueError("Box too small to fit one patch; enlarge.")
    return slice(i0, i0+Hc), slice(j0, j0+Wc)

def norm_gc_tp06(x):
    return (x - graphcast_min["tp06"]) / (graphcast_max["tp06"] - graphcast_min["tp06"])

def denorm_aorc_precip(y):
    return y * (aorc_max["APCP_surface"] - aorc_min["APCP_surface"]) + aorc_min["APCP_surface"]

def panel_triplet(da_true, da_pred, t, vmax=None, diff_lim=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    da_true.isel(time=t).plot.imshow(ax=axes[0], vmin=0, vmax=vmax, add_colorbar=True);  axes[0].set_title(f"Truth t+{t}h"); axes[0].set_xlabel(""); axes[0].set_ylabel("")
    da_pred.isel(time=t).plot.imshow(ax=axes[1], vmin=0, vmax=vmax, add_colorbar=True);  axes[1].set_title(f"Pred t+{t}h");  axes[1].set_xlabel(""); axes[1].set_ylabel("")
    (da_pred.isel(time=t)-da_true.isel(time=t)).plot.imshow(ax=axes[2],
        vmin=(-diff_lim if diff_lim is not None else None),
        vmax=( diff_lim if diff_lim is not None else None),
        add_colorbar=True); axes[2].set_title("Pred − Truth"); axes[2].set_xlabel(""); axes[2].set_ylabel("")
    return fig


# ========== STEP 0: Config ==========
GRAPHCAST_BASE = "/scratch/08105/ms86336/graphcast"
AORC_BASE = "/scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa"

YEARS = [2018, 2019, 2020]

# Vars (inputs from GraphCast) -> (targets in AORC), channel-aligned
VAR_MAP = {
    "u10m": "UGRD_10maboveground",
    "v10m": "VGRD_10maboveground",
    "t2m":  "TMP_2maboveground",
    "tp06": "APCP_surface",
}
GC_VARS = list(VAR_MAP.keys())
AORC_VARS = list(VAR_MAP.values())

print(GC_VARS)

print(AORC_VARS)

# # ---- config: pick ONE GraphCast file to start ----
# gc_path = Path("/scratch/08105/ms86336/graphcast/2020/graphcast_2020_07_03.nc")

# # 1) derive the corresponding AORC path from the GraphCast filename (YYYY_MM_DD -> YYYYMMDD)
# m = re.search(r"graphcast_(\d{4})_(\d{2})_(\d{2})\.nc$", gc_path.name)
# assert m, f"Unexpected GraphCast filename: {gc_path.name}"
# yyyy, mm, dd = m.groups()
# aorc_path = Path(f"/scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/noaa_aorc_usa_{yyyy}_day_{yyyy}{mm}{dd}.nc")
# print("GC file:", gc_path)
# print("AORC file:", aorc_path)

# # 2) open both datasets
# ds_gc = xr.open_dataset(gc_path, engine="netcdf4")
# ds_aorc = xr.open_dataset(aorc_path, engine="netcdf4")

# # 3) standardize coord names to lat/lon if needed (inline, no helper)
# if "latitude" in ds_gc.dims: ds_gc = ds_gc.rename({"latitude": "lat"})
# if "longitude" in ds_gc.dims: ds_gc = ds_gc.rename({"longitude": "lon"})
# if "latitude" in ds_gc.variables: ds_gc = ds_gc.rename({"latitude": "lat"})
# if "longitude" in ds_gc.variables: ds_gc = ds_gc.rename({"longitude": "lon"})

# if "latitude" in ds_aorc.dims: ds_aorc = ds_aorc.rename({"latitude": "lat"})
# if "longitude" in ds_aorc.dims: ds_aorc = ds_aorc.rename({"longitude": "lon"})
# if "latitude" in ds_aorc.variables: ds_aorc = ds_aorc.rename({"latitude": "lat"})
# if "longitude" in ds_aorc.variables: ds_aorc = ds_aorc.rename({"longitude": "lon"})


# # 4) select just the four variables
# GC_VARS = ["u10m", "v10m", "t2m", "tp06"]
# AORC_VARS = ["UGRD_10maboveground", "VGRD_10maboveground", "TMP_2maboveground", "APCP_surface"]

# missing_gc = [v for v in GC_VARS if v not in ds_gc.variables]
# missing_aorc = [v for v in AORC_VARS if v not in ds_aorc.variables]
# if missing_gc: print("[warn] missing in GC:", missing_gc)
# if missing_aorc: print("[warn] missing in AORC:", missing_aorc)

# ds_gc = ds_gc[[v for v in GC_VARS if v in ds_gc.variables]]
# ds_aorc = ds_aorc[[v for v in AORC_VARS if v in ds_aorc.variables]]

# # 5) if GraphCast vars have a "history" dim, pick the first lead
# for v in ds_gc.data_vars:
#     if "history" in ds_gc[v].dims:
#         ds_gc[v] = ds_gc[v].isel(history=0)
# print(ds_gc)

# print(ds_aorc)

# GC file: /scratch/08105/ms86336/graphcast/2018/graphcast_2018_07_03.nc
# AORC file: /scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/noaa_aorc_usa_2018_day_20180703.nc
# <xarray.Dataset> Size: 1GB
# Dimensions:  (time: 61, history: 1, lat: 721, lon: 1440)
# Coordinates:
#   * lat      (lat) float64 6kB -90.0 -89.75 -89.5 -89.25 ... 89.5 89.75 90.0
#   * lon      (lon) float64 12kB 0.0 0.25 0.5 0.75 ... 359.0 359.2 359.5 359.8
#   * time     (time) datetime64[ns] 488B 2018-07-03 ... 2018-07-18
# Dimensions without coordinates: history
# Data variables:
#     u10m     (time, history, lat, lon) float32 253MB ...
#     v10m     (time, history, lat, lon) float32 253MB ...
#     t2m      (time, history, lat, lon) float32 253MB ...
#     tp06     (time, history, lat, lon) float32 253MB ...
# -bash: __vsc_prompt_cmd_original: command not found
# (base) c612-101[gh](1037)$ apptainer exec --nv ../ai_land_model/apptainer_al_land.sif python train.py 
# INFO:    gocryptfs not found, will not be able to use gocryptfs
# ['u10m', 'v10m', 't2m', 'tp06']
# ['UGRD_10maboveground', 'VGRD_10maboveground', 'TMP_2maboveground', 'APCP_surface']
# GC file: /scratch/08105/ms86336/graphcast/2018/graphcast_2018_07_03.nc
# AORC file: /scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/noaa_aorc_usa_2018_day_20180703.nc
# <xarray.Dataset> Size: 1GB
# Dimensions:  (time: 61, history: 1, lat: 721, lon: 1440)
# Coordinates:
#   * lat      (lat) float64 6kB -90.0 -89.75 -89.5 -89.25 ... 89.5 89.75 90.0
#   * lon      (lon) float64 12kB 0.0 0.25 0.5 0.75 ... 359.0 359.2 359.5 359.8
#   * time     (time) datetime64[ns] 488B 2018-07-03 ... 2018-07-18
# Dimensions without coordinates: history
# Data variables:
#     u10m     (time, history, lat, lon) float32 253MB ...
#     v10m     (time, history, lat, lon) float32 253MB ...
#     t2m      (time, history, lat, lon) float32 253MB ...
#     tp06     (time, history, lat, lon) float32 253MB ...
# <xarray.Dataset> Size: 27GB
# Dimensions:              (time: 24, lat: 4201, lon: 8401)
# Coordinates:
#   * lat                  (lat) float64 34kB 20.0 20.01 20.02 ... 54.99 55.0
#   * lon                  (lon) float64 67kB -130.0 -130.0 ... -60.01 -60.0
#   * time                 (time) datetime64[ns] 192B 2018-07-03 ... 2018-07-03...
# Data variables:
#     UGRD_10maboveground  (time, lat, lon) float64 7GB ...
#     VGRD_10maboveground  (time, lat, lon) float64 7GB ...
#     TMP_2maboveground    (time, lat, lon) float64 7GB ...
#     APCP_surface         (time, lat, lon) float64 7GB ...
# -bash: __vsc_prompt_cmd_original: command not found
# (base) c612-101[gh](1038)$ apptainer exec --nv ../ai_land_model/apptainer_al_land.sif python train.py 
# INFO:    gocryptfs not found, will not be able to use gocryptfs
# ['u10m', 'v10m', 't2m', 'tp06']
# ['UGRD_10maboveground', 'VGRD_10maboveground', 'TMP_2maboveground', 'APCP_surface']
# GC file: /scratch/08105/ms86336/graphcast/2018/graphcast_2018_07_03.nc
# AORC file: /scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/noaa_aorc_usa_2018_day_20180703.nc
# <xarray.Dataset> Size: 1GB
# Dimensions:  (time: 61, lat: 721, lon: 1440)
# Coordinates:
#   * lat      (lat) float64 6kB -90.0 -89.75 -89.5 -89.25 ... 89.5 89.75 90.0
#   * lon      (lon) float64 12kB 0.0 0.25 0.5 0.75 ... 359.0 359.2 359.5 359.8
#   * time     (time) datetime64[ns] 488B 2018-07-03 ... 2018-07-18
# Data variables:
#     u10m     (time, lat, lon) float32 253MB ...
#     v10m     (time, lat, lon) float32 253MB ...
#     t2m      (time, lat, lon) float32 253MB ...
#     tp06     (time, lat, lon) float32 253MB ...
# <xarray.Dataset> Size: 27GB
# Dimensions:              (time: 24, lat: 4201, lon: 8401)
# Coordinates:
#   * lat                  (lat) float64 34kB 20.0 20.01 20.02 ... 54.99 55.0
#   * lon                  (lon) float64 67kB -130.0 -130.0 ... -60.01 -60.0
#   * time                 (time) datetime64[ns] 192B 2018-07-03 ... 2018-07-03...
# Data variables:
#     UGRD_10maboveground  (time, lat, lon) float64 7GB ...
#     VGRD_10maboveground  (time, lat, lon) float64 7GB ...
#     TMP_2maboveground    (time, lat, lon) float64 7GB ...
#     APCP_surface         (time, lat, lon) float64 7GB ...


# 6) ***CROP GraphCast to AORC domain***
# AORC is on lon [-130, -60], lat [20, 55] (you can also read these from the file)
# a_lat_min = float(ds_aorc["lat"].min().values)
# a_lat_max = float(ds_aorc["lat"].max().values)
# a_lon_min = float(ds_aorc["lon"].min().values)
# a_lon_max = float(ds_aorc["lon"].max().values)

# # GraphCast lon is 0..360; convert to -180..180 and sort so slicing works
# ds_gc = ds_gc.assign_coords(lon=((ds_gc["lon"] + 180) % 360) - 180).sortby("lon")

# # Ensure latitude is ascending for slice()
# if ds_gc["lat"][0] > ds_gc["lat"][-1]:
#     ds_gc = ds_gc.sortby("lat")

# # Now crop GraphCast to AORC bbox (this massively shrinks memory)
# ds_gc = ds_gc.sel(lat=slice(a_lat_min, a_lat_max), lon=slice(a_lon_min, a_lon_max))

# print("GC after crop:", {d: ds_gc.dims[d] for d in ds_gc.dims})
# print("AORC dims    :", {d: ds_aorc.dims[d] for d in ds_aorc.dims})

# # 7) Interpolate GraphCast to AORC grid (bilinear). Now we’re only interpolating the US box.
# ds_gc = ds_gc.interp(lat=ds_aorc["lat"], lon=ds_aorc["lon"], kwargs={"fill_value": "extrapolate"})

# print(ds_gc)
# AORC dims    : {'time': 24, 'lat': 4201, 'lon': 8401}
# <xarray.Dataset> Size: 69GB
# Dimensions:  (time: 61, lat: 4201, lon: 8401)
# Coordinates:
#   * time     (time) datetime64[ns] 488B 2018-07-03 ... 2018-07-18
#   * lat      (lat) float64 34kB 20.0 20.01 20.02 20.02 ... 54.98 54.99 55.0
#   * lon      (lon) float64 67kB -130.0 -130.0 -130.0 ... -60.02 -60.01 -60.0
# Data variables:
#     u10m     (time, lat, lon) float64 17GB 1.144 1.15 1.156 ... 1.656 1.678
#     v10m     (time, lat, lon) float64 17GB -2.673 -2.657 ... -2.423 -2.436
#     t2m      (time, lat, lon) float64 17GB 296.3 296.3 296.3 ... 279.8 279.8
#     tp06     (time, lat, lon) float64 17GB nan nan nan ... 0.006911 0.006918
# -bash: __vsc_prompt_cmd_original: command not found

# # --- choose 0–72h window from GraphCast (you already have this) ---
# gc_t0 = ds_gc["time"].values[0]
# gc_t72 = gc_t0 + np.timedelta64(72, "h")
# ds_gc = ds_gc.sel(time=slice(gc_t0, gc_t72))

# # If you still have a single-day AORC open above, free the handle:
# try:
#     ds_aorc.close()
# except Exception:
#     pass



# d0 = datetime(int(yyyy), int(mm), int(dd))
# aorc_files = []
# for i in range(3):
#     di = d0 + timedelta(days=i)
#     ymd = di.strftime("%Y%m%d")
#     aorc_files.append(
#         f"/scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/noaa_aorc_usa_{di.year}_day_{ymd}.nc"
#     )

# # Open each file (try netcdf4 then h5netcdf), then combine.
# # Avoid parallel=True here; it can tickle HDF5 issues.
# dsets = []
# for p in aorc_files:
#     ok = False
#     for eng in ("netcdf4", "h5netcdf"):
#         try:
#             ds_i = xr.open_dataset(p, engine=eng, chunks={"time": 24})
#             _ = ds_i["time"].isel(time=0).values  # touch to validate
#             dsets.append(ds_i)
#             print(f"[ok] {Path(p).name} via {eng}")
#             ok = True
#             break
#         except Exception as e:
#             last_err = e
#     if not ok:
#         print(f"[skip] {Path(p).name}: {last_err}")

# if not dsets:
#     raise RuntimeError("No readable AORC files for day0..day2")

# ds_aorc = xr.combine_by_coords(dsets, combine_attrs="override")

# # standardize coords if needed
# if "latitude" in ds_aorc.dims: ds_aorc = ds_aorc.rename({"latitude": "lat"})
# if "longitude" in ds_aorc.dims: ds_aorc = ds_aorc.rename({"longitude": "lon"})
# if "latitude" in ds_aorc.variables: ds_aorc = ds_aorc.rename({"latitude": "lat"})
# if "longitude" in ds_aorc.variables: ds_aorc = ds_aorc.rename({"longitude": "lon"})

# # keep only your vars + chunk
# AORC_VARS = ["UGRD_10maboveground", "VGRD_10maboveground", "TMP_2maboveground", "APCP_surface"]
# ds_aorc = ds_aorc[AORC_VARS].chunk({"time": 24, "lat": 512, "lon": 512})

# print(ds_gc)

# print(ds_aorc)


# class ncDataset(Dataset):
#     def __init__(self, data, targets):
#         self.data = data
#         self.targets = targets

#     def __getitem__(self, index):
#         x = torch.from_numpy(self.data[index]).unsqueeze(0)
#         y = torch.from_numpy(self.targets[index]).unsqueeze(0)
#         # x = self.data[index]
#         # y = self.targets[index]
#         # x = x.to(dtype=torch.float32)
#         # y = y.to(dtype=torch.float32)
#         return x, y

#     def __len__(self):
#         return len(self.data)

class ncDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data          # (N, 12, 128, 128)
        self.targets = targets    # (N, 72, 128, 128)

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).to(torch.float32)    # (12,128,128)
        y = torch.from_numpy(self.targets[index]).to(torch.float32) # (72,128,128)
        return x, y

    def __len__(self):
        return len(self.data)

# x_train = ds_gc.tp06.values[1:,:,:].astype(np.float32)
# y_train = ds_aorc.APCP_surface.values[:,:].astype(np.float32)
# print('x_train.shape', x_train.shape)
# print('y_train.shape', y_train.shape)



# --- your original 2D helpers (unchanged) ---
def unpatchify(patches, img_shape):
    patch_size = patches.shape[1]
    assert patches.shape[0] == (img_shape[0] // patch_size) * (img_shape[1] // patch_size), "Patches and image shape are not compatible"
    img = np.zeros(img_shape, dtype=patches.dtype)
    patch_idx = 0
    for i in range(0, img_shape[0], patch_size):
        for j in range(0, img_shape[1], patch_size):
            img[i:i + patch_size, j:j + patch_size] = patches[patch_idx]
            patch_idx += 1
    return img

def patchify(img, patch_size):
    H, W = img.shape
    # only take FULL patches so all patches are exactly (patch_size, patch_size)
    Hc = H - (H % patch_size)
    Wc = W - (W % patch_size)
    return np.array([
        img[i:i+patch_size, j:j+patch_size]
        for i in range(0, Hc, patch_size)
        for j in range(0, Wc, patch_size)
    ], dtype=img.dtype)

# --- wrappers that operate over time and stack to (N, T, ph, pw) ---
def patchify_time_stack(arr_3d, patch_size):
    """
    arr_3d: (T, H, W)
    returns:
      patches: (N, T, patch_size, patch_size)
      meta: dict for unpatchify_time_stack()
    """
    T, H, W = arr_3d.shape
    Hc = H - (H % patch_size)
    Wc = W - (W % patch_size)

    # patchify each time slice -> (N, ph, pw), ensure N is consistent
    per_t = [patchify(arr_3d[t, :Hc, :Wc], patch_size) for t in range(T)]
    N = per_t[0].shape[0]
    for t in range(1, T):
        assert per_t[t].shape[0] == N, "Inconsistent patch counts across time"

    # stack into (N, T, ph, pw)
    patches = np.stack(per_t, axis=1)   # (N, T, ph, pw)
    meta = {"T": T, "H": H, "W": W, "Hc": Hc, "Wc": Wc, "patch": patch_size}
    return patches, meta

def unpatchify_time_stack(patches, meta):
    """
    patches: (N, T, ph, pw)
    returns:
      arr_3d: (T, H, W) (cropped back to Hc,Wc inside, then placed in H,W)
    """
    N, T, ph, pw = patches.shape
    assert T == meta["T"] and ph == meta["patch"] and pw == meta["patch"]
    Hc, Wc = meta["Hc"], meta["Wc"]

    out = np.zeros((T, meta["H"], meta["W"]), dtype=patches.dtype)
    for t in range(T):
        # take patches for this time: (N, ph, pw) and use your 2D unpatchify
        recon_2d = unpatchify(patches[:, t, :, :], (Hc, Wc))
        out[t, :Hc, :Wc] = recon_2d
    return out

# --- NaN filtering across time (strict: drop if ANY NaN in x OR y) ---
def filter_patches_no_nan(x_patches, y_patches):
    """
    x_patches: (N, Tx, ph, pw)
    y_patches: (N, Ty, ph, pw)
    keep only patch indices where BOTH have no NaNs across all time slices
    returns filtered (x_patches, y_patches) and kept indices
    """
    x_ok = ~np.isnan(x_patches).any(axis=(1,2,3))
    y_ok = ~np.isnan(y_patches).any(axis=(1,2,3))
    keep = x_ok & y_ok
    idxs = np.nonzero(keep)[0]
    return x_patches[keep], y_patches[keep], idxs


# Given:
# x_train.shape == (12, 4201, 8401)
# y_train.shape == (72, 4201, 8401)

# ps = 128

# # 1) Make (num_patches, T, 128, 128)
# x_patches, x_meta = patchify_time_stack(x_train, ps)   # -> (Nx, 12, 128, 128)
# y_patches, y_meta = patchify_time_stack(y_train, ps)   # -> (Ny, 72, 128, 128)
# assert x_patches.shape[0] == y_patches.shape[0], "Patch counts must match (same H,W & ps)."

# print('x_patches.shape', x_patches.shape)
# print('y_patches.shape', y_patches.shape)

# # 2) Drop any patch that has NaNs in x OR y (across all time)
# x_patches, y_patches, kept = filter_patches_no_nan(x_patches, y_patches)
# print("kept patches:", x_patches.shape[0])

# print(x_patches.shape, y_patches.shape)

# No need to unpatchify as things work without it

# # 3) (optional) Reconstruct back to (T,H,W)
# #    Note: unpatchify_time_stack reconstructs only the full multiples region (Hc,Wc)
# x_recon = unpatchify_time_stack(x_patches, x_meta)
# y_recon = unpatchify_time_stack(y_patches, y_meta)

# x_patches = (x_patches - graphcast_min['tp06'])/ (graphcast_max['tp06'] - graphcast_min['tp06'])
# y_patches = (y_patches - aorc_min['APCP_surface'])/ (aorc_max['APCP_surface'] - aorc_min['APCP_surface'])


# train_dataset = ncDataset(x_patches, y_patches)

from datetime import datetime, timedelta

# --------- BIG DATE LOOP: 2018-01-01 → 2020-12-31 ---------
#start_date = datetime(2018, 1, 1)
#start_date = datetime(2018, 4, 16)
start_date = datetime(2019, 6, 17)
end_date   = datetime(2020, 12, 31)


upscale_factor = 1
batch_size = 1
in_channels = 12
out_channels = 72
height, width = 128, 128  # Adjust height and width as needed

# Initialize the SRResNet model with the given upscale factor
netG = SRResNet(in_channels=in_channels, out_channels=out_channels, upscale=upscale_factor).cuda()
# generator = Generator()
# netD = Discriminator().cuda()
netD = Discriminator(in_channels=72).cuda()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
mse = nn.MSELoss()

optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
optimizerD = optim.Adam(netD.parameters(), lr=1e-4)
# input_tensor = torch.rand((batch_size, in_channels, height, width)).cuda()
# ---- after creating netG/netD and (optionally) moving to CUDA ----
ckpt_path = "checkpoints/netG-best.pth"   # or your actual weight file
map_loc   = "cuda" if torch.cuda.is_available() else "cpu"

try:
    state_dict = torch.load(ckpt_path, map_location=map_loc)
    netG.load_state_dict(state_dict, strict=True)  # strict=False if you changed layers/shapes
    print(f"[resume] Loaded generator weights from {ckpt_path}")
except Exception as e:
    print(f"[warn] Could not load {ckpt_path}: {e}")

is_train = True
# Pre-train generator using only MSE loss
n_epoch_pretrain = 2

optimizerG = optim.Adam(netG.parameters())

def gaussian(window_size, sigma):
    gauss = torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)])
    gauss = torch.exp(gauss)
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, val_range=1):
    (_, channel, height, width) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# # Simple PSNR calculation function
# def psnr(img1, img2):
#     mse = F.mse_loss(img1, img2)
#     if mse == 0:
#         return float('inf')
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))

import torch
import torch.nn.functional as F

def psnr(img1, img2, max_val=1.0, eps=1e-12):
    # scalar tensor
    mse = F.mse_loss(img1, img2)
    # avoid log(0)
    mse = torch.clamp(mse, min=eps)
    max_val_t = torch.tensor(max_val, device=img1.device, dtype=img1.dtype)
    # 20*log10(MAX_I) - 10*log10(MSE)
    return 20.0 * torch.log10(max_val_t) - 10.0 * torch.log10(mse)


d = start_date

wandb.login(key="70f85253c59220a4439123cc3c97280ece560bf5")  # Replace with your actual key or use `wandb.login()` interactively

run = wandb.init(
    project="meteogan2",           # <- change if you like
    name=f"srgan-gc2aorc-precip",# unique per start file/day
    config={
        "in_channels": in_channels,
        "out_channels": out_channels,
        "patch_size": 128,
        "lrG": 1e-4, "lrD": 1e-4,
        "gc_norm_min": graphcast_min["tp06"],
        "gc_norm_max": graphcast_max["tp06"],
        "aorc_norm_min": aorc_min["APCP_surface"],
        "aorc_norm_max": aorc_max["APCP_surface"],
    },
    save_code=True,  # uploads this training script for provenance
)
wandb.watch(netG, log="all", log_freq=100)  # gradients & param histograms
wandb.watch(netD, log="all", log_freq=100)

while d <= end_date:
    all_x_patches = []
    all_y_patches = []
    yyyy = f"{d.year:04d}"
    mm   = f"{d.month:02d}"
    dd   = f"{d.day:02d}"

    gc_path   = Path(f"{GRAPHCAST_BASE}/{yyyy}/graphcast_{yyyy}_{mm}_{dd}.nc")
    aorc_path = Path(f"{AORC_BASE}/noaa_aorc_usa_{yyyy}_day_{yyyy}{mm}{dd}.nc")

    if not gc_path.exists() or not aorc_path.exists():
        print(f"[skip] missing file(s) {yyyy}-{mm}-{dd}")
        d += timedelta(days=1)
        continue

    try:
        ds_gc = xr.open_dataset(gc_path, engine="netcdf4")
        ds_aorc = xr.open_dataset(aorc_path, engine="netcdf4")
    except Exception as e:
        print(f"[skip] open error {yyyy}-{mm}-{dd}: {e}")
        d += timedelta(days=1)
        continue

    # --- coordinate standardization (inline, no helpers required) ---
    if "latitude" in ds_gc.dims: ds_gc = ds_gc.rename({"latitude":"lat"})
    if "longitude" in ds_gc.dims: ds_gc = ds_gc.rename({"longitude":"lon"})
    if "latitude" in ds_aorc.dims: ds_aorc = ds_aorc.rename({"latitude":"lat"})
    if "longitude" in ds_aorc.dims: ds_aorc = ds_aorc.rename({"longitude":"lon"})

    # subset variables
    GC_VARS   = ["u10m","v10m","t2m","tp06"]
    AORC_VARS = ["UGRD_10maboveground","VGRD_10maboveground","TMP_2maboveground","APCP_surface"]
    ds_gc   = ds_gc[[v for v in GC_VARS   if v in ds_gc.variables]]
    ds_aorc = ds_aorc[[v for v in AORC_VARS if v in ds_aorc.variables]]

    # drop GraphCast history lead if present
    for v in ds_gc.data_vars:
        if "history" in ds_gc[v].dims:
            ds_gc[v] = ds_gc[v].isel(history=0)

    # AORC bbox
    a_lat_min = float(ds_aorc["lat"].min().values)
    a_lat_max = float(ds_aorc["lat"].max().values)
    a_lon_min = float(ds_aorc["lon"].min().values)
    a_lon_max = float(ds_aorc["lon"].max().values)

    # wrap/sort GC lon, sort lat ascend, crop to AORC box
    ds_gc = ds_gc.assign_coords(lon=((ds_gc["lon"] + 180) % 360) - 180).sortby("lon")
    if ds_gc["lat"][0] > ds_gc["lat"][-1]:
        ds_gc = ds_gc.sortby("lat")
    ds_gc = ds_gc.sel(lat=slice(a_lat_min, a_lat_max), lon=slice(a_lon_min, a_lon_max))

    # bilinear interp GC -> AORC grid
    ds_gc = ds_gc.interp(lat=ds_aorc["lat"], lon=ds_aorc["lon"], kwargs={"fill_value":"extrapolate"})

    # choose 0–72h window from GC (t0..t0+72h)
    gc_t0  = ds_gc["time"].values[0]
    gc_t72 = gc_t0 + np.timedelta64(72, "h")
    ds_gc  = ds_gc.sel(time=slice(gc_t0, gc_t72))

    # reopen AORC for day 0..2 (hourly) to match 72h target horizon
    try:
        ds_aorc.close()
    except:  # not critical
        pass

    d0 = datetime(int(yyyy), int(mm), int(dd))
    aorc_files = []
    for i in range(3):
        di  = d0 + timedelta(days=i)
        ymd = di.strftime("%Y%m%d")
        aorc_files.append(f"{AORC_BASE}/noaa_aorc_usa_{di.year}_day_{ymd}.nc")

    dsets = []
    last_err = None
    for p in aorc_files:
        ok = False
        for eng in ("netcdf4","h5netcdf"):
            try:
                ds_i = xr.open_dataset(p, engine=eng, chunks={"time":24})
                _ = ds_i["time"].isel(time=0).values
                dsets.append(ds_i); ok = True; break
            except Exception as e:
                last_err = e
        if not ok:
            print(f"[skip] {Path(p).name}: {last_err}")
    if not dsets:
        print(f"[skip] no AORC trio for {yyyy}-{mm}-{dd}")
        d += timedelta(days=1)
        continue

    ds_aorc = xr.combine_by_coords(dsets, combine_attrs="override")
    if "latitude" in ds_aorc.dims: ds_aorc = ds_aorc.rename({"latitude":"lat"})
    if "longitude" in ds_aorc.dims: ds_aorc = ds_aorc.rename({"longitude":"lon"})
    AORC_VARS = ["UGRD_10maboveground","VGRD_10maboveground","TMP_2maboveground","APCP_surface"]
    ds_aorc = ds_aorc[AORC_VARS].chunk({"time":24,"lat":512,"lon":512})

    # --------- extract 12×GC tp06 and 72×AORC precip, patchify, normalize ---------
    # X: 12 steps (skip t0 so 1..12) ; Y: 72 steps
    try:
        x_train = ds_gc.tp06.values[1:,:,:].astype(np.float32)      # (12,H,W)
        y_train = ds_aorc.APCP_surface.values[:,:].astype(np.float32)  # (72,H,W)
    except Exception as e:
        print(f"[skip] var read error {yyyy}-{mm}-{dd}: {e}")
        try:
            ds_gc.close(); ds_aorc.close()
        except: pass
        d += timedelta(days=1)
        continue

    ps = 128
    # patchify over time (T,H,W)->(N,T,ps,ps)
    # (using your patchify_time_stack, filter_patches_no_nan from above)
    x_patches_d, _ = patchify_time_stack(x_train, ps)
    y_patches_d, _ = patchify_time_stack(y_train, ps)

    # drop NaN patches
    x_patches_d, y_patches_d, kept = filter_patches_no_nan(x_patches_d, y_patches_d)
    if x_patches_d.shape[0] == 0:
        print(f"[skip] all-NaN after patchify {yyyy}-{mm}-{dd}")
        try:
            ds_gc.close(); ds_aorc.close()
        except: pass
        d += timedelta(days=1)
        continue

    # normalize
    x_patches_d = (x_patches_d - graphcast_min['tp06']) / (graphcast_max['tp06'] - graphcast_min['tp06'])
    y_patches_d = (y_patches_d - aorc_min['APCP_surface']) / (aorc_max['APCP_surface'] - aorc_min['APCP_surface'])

    # # accumulate
    all_x_patches.append(x_patches_d)   # list of (N,12,128,128)
    all_y_patches.append(y_patches_d)   # list of (N,72,128,128)

    # close datasets to free memory
    try:
        ds_gc.close(); ds_aorc.close()
    except: pass

    print(f"[ok] {yyyy}-{mm}-{dd} kept patches: {x_patches_d.shape[0]}")
    d += timedelta(days=1)

    # --------- CONCAT EVERYTHING to feed your existing DataLoader/training ---------
    if len(all_x_patches)==0:
        raise RuntimeError("No training patches found over 2018-01-01..2020-12-31")

    x_patches = np.concatenate(all_x_patches, axis=0)
    y_patches = np.concatenate(all_y_patches, axis=0)

    print('x_patches.shape (ALL DAYS)', x_patches.shape)  # (N_total,12,128,128)
    print('y_patches.shape (ALL DAYS)', y_patches.shape)  # (N_total,72,128,128)
    train_dataset = ncDataset(x_patches, y_patches)


    # Split into train/val
    val_split = 0.1
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=16, shuffle=False)






# # Update Generator to handle 1 channel input and output
# generator.conv1 = nn.Sequential(
#     nn.Conv2d(1, 64, kernel_size=9, padding=4),
#     nn.PReLU()
# )
# generator.upsample = nn.Sequential(
#     UpsampleBLock(64, 2),
#     UpsampleBLock(64, 2),
#     nn.Conv2d(64, 1, kernel_size=9, padding=4)
# )

# Update Discriminator to handle 1 channel input
# netD.net[0] = nn.Conv2d(out_channels, 64, kernel_size=3, padding=1).cuda()

# # Test the Generator
# gen_output = netG(input_tensor)
# print("Generator input size:", input_tensor.size())
# print("Generator output size:", gen_output.size())

# # Test the Discriminator
# disc_output = netD(gen_output)
# print("Discriminator output size:", disc_output.size())

# print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
	
# print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))


    if is_train:
        for epoch in range(1, n_epoch_pretrain + 1):	
            train_bar = tqdm(train_dataloader)
            
            netG.train()
            
            cache = {'g_loss': 0}
            
            for lowres, real_img_hr in train_bar:
                if torch.cuda.is_available():
                    real_img_hr = real_img_hr.cuda()
                    
                if torch.cuda.is_available():
                    lowres = lowres.cuda()
                    
                fake_img_hr = netG(lowres)

                # Train G
                netG.zero_grad()
                
                image_loss = mse(fake_img_hr, real_img_hr)
                cache['g_loss'] += image_loss
                
                image_loss.backward()
                optimizerG.step()

                # Print information by tqdm
                train_bar.set_description(desc='[%d/%d] Loss_G: %.4f' % (epoch, n_epoch_pretrain, image_loss))





# Assume these functions are defined somewhere else in the code
# compute_gradient_penalty, get_grads_D_WAN, get_grads_G, mse, ssim, psnr



    n_epochs = 20
    best_ssim = 0  # To keep track of the best SSIM score
# Initialize wandb


    if is_train:

        for epoch in range(n_epochs):
            train_bar = tqdm(train_dataloader)
            val_bar = tqdm(val_dataloader)
            
            netG.train()
            netD.train()
            
            cache = {'mse_loss': 0, 'adv_loss': 0, 'g_loss': 0, 'd_loss': 0, 'ssim': 0, 'psnr': 0, 'd_top_grad': 0, 'd_bot_grad': 0, 'g_top_grad': 0, 'g_bot_grad': 0}
            
            for lowres, real_img_hr in train_bar:
                if torch.cuda.is_available():
                    real_img_hr = real_img_hr.cuda()
                    lowres = lowres.cuda()
                    
                fake_img_hr = netG(lowres)
                
                # Train D
                netD.zero_grad()
                
                logits_real = netD(real_img_hr).mean()
                logits_fake = netD(fake_img_hr).mean()
                gradient_penalty = compute_gradient_penalty(netD, real_img_hr, fake_img_hr)
                
                d_loss = logits_fake - logits_real + 10 * gradient_penalty

                
                cache['d_loss'] += d_loss.item()
                
                d_loss.backward(retain_graph=True)
                optimizerD.step()
                # Clean up memory
                

                
                dtg, dbg = get_grads_D_WAN(netD)
                cache['d_top_grad'] += dtg
                cache['d_bot_grad'] += dbg
                wandb.log({
                    "train/d_loss": float(d_loss.item()),
                    "train/d_top_grad": float(dtg),
                    "train/d_bot_grad": float(dbg),
                    "train/epoch": epoch
                })
                # Train G
                netG.zero_grad()
                
                image_loss = mse(fake_img_hr, real_img_hr)
                adversarial_loss = -1 * netD(fake_img_hr).mean()
                
                g_loss = image_loss + 1e-3 * adversarial_loss


                cache['mse_loss'] += image_loss.item()
                cache['adv_loss'] += adversarial_loss.item()
                cache['g_loss'] += g_loss.item()

                g_loss.backward()
                optimizerG.step()
                
                gtg, gbg = get_grads_G(netG)
                cache['g_top_grad'] += gtg
                cache['g_bot_grad'] += gbg
                wandb.log({
                    "train/g_loss": float(g_loss.item()),
                    "train/g_mse": float(image_loss.item()),
                    "train/g_adv": float(adversarial_loss.item()),
                    "train/g_top_grad": float(gtg),
                    "train/g_bot_grad": float(gbg),
                    "train/epoch": epoch
                })
                del lowres, real_img_hr, fake_img_hr  # Delete variables
                torch.cuda.empty_cache()  # Clear CUDA memory cache
                # Print information by tqdm
                train_bar.set_description(desc='[%d/%d] D grads:(%f, %f) G grads:(%f, %f) Loss_D: %.4f Loss_G: %.4f = %.4f + %.4f' % (epoch, n_epochs, dtg, dbg, gtg, gbg, d_loss, g_loss, image_loss, adversarial_loss))
            
            # Evaluate on validation set
            netG.eval()
            val_ssim = 0
            val_psnr = 0
            with torch.no_grad():
                for lowres, real_img_hr in val_bar:
                    if torch.cuda.is_available():
                        real_img_hr = real_img_hr.cuda()
                        lowres = lowres.cuda()
                        
                    fake_img_hr = netG(lowres)
                    val_ssim += ssim(fake_img_hr, real_img_hr).item()
                    val_psnr += psnr(fake_img_hr, real_img_hr).item()
                    del lowres, real_img_hr, fake_img_hr  # Delete variables
                    torch.cuda.empty_cache()  # Clear CUDA memory cache
                    
                
            val_ssim /= len(val_dataloader)
            val_psnr /= len(val_dataloader)
            wandb.log({
                "valid/ssim": float(val_ssim),
                "valid/psnr": float(val_psnr),
                "epoch": epoch
            })
            
            # Save the best model
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save(netG.state_dict(), 'best_netG.pth')
                torch.save(netG.state_dict(), "checkpoints/netG-best.pth")
                best_art = wandb.Artifact("srgan-gc2aorc-best", type="model",
                                        metadata={"selection":"best_by_ssim","who":"netG"})
                best_art.add_file("checkpoints/netG-best.pth")
                wandb.log_artifact(best_art, aliases=["best"])
            
            print(f'Epoch [{epoch}/{n_epochs}] Validation SSIM: {val_ssim:.4f} PSNR: {val_psnr:.4f}')
