#from google.colab import drive
#drive.mount('/content/drive')

import xarray as xr
import sys
import os
#!ls /content/drive/MyDrive/
#!mkdir /content/drive/MyDrive/meteogan
#!ls /content/drive/MyDrive/meteogan -lrt

download_data = False

#!git clone https://github.com/scotthosking/get-station-data.git
#!mv get-station-data/* .
#!pip install -v -e .

import pandas as pd
from get_station_data import ghcnd
from get_station_data.util import nearest_stn

#%matplotlib inline

stn_md = ghcnd.get_stn_metadata()

delhi_lon_lat = 77.1025, 28.7041
austin_lon_lat =  -97.7431,  30.2672
my_stns = nearest_stn(stn_md,
                        delhi_lon_lat[0], delhi_lon_lat[1],
                        n_neighbours=100)

df_ = ghcnd.get_data(my_stns)

df__ = df_[df_['element']=='PRCP'].copy()
df_camp_mabry = df__.copy() #  df__[df__['name']=='TX AUSTIN-CAMP MABRY'].copy()
# df_camp_mabry['month'] = df_camp_mabry['month'].astype(str).str.zfill(2)
# df_camp_mabry['day'] = df_camp_mabry['day'].astype(str).str.zfill(2)
# df_camp_mabry['datetime'] = df_camp_mabry['year'].astype(str)+df_camp_mabry['month']+df_camp_mabry['day']
df_camp_mabry['datetime'] = pd.to_datetime(df_camp_mabry['date'], format="%Y-%m-%d")
df_camp_mabry = df_camp_mabry.set_index(['datetime'])
cols = ['value']
for name in df_camp_mabry.name.unique():
    print(name)
    #df_camp_mabry[cols][df_camp_mabry['name']=='TX AUSTIN-CAMP MABRY'].plot()
    df_camp_mabry[cols][df_camp_mabry['name']==name].plot()

# Found that nan might not be a good choice, rather we should choose the stations which
# have a long term data
stations_with_long_term_data = []
for name in df_camp_mabry.name.unique():
#    print(df_camp_mabry[df_camp_mabry['name']==name].date[0].year, df_camp_mabry[df_camp_mabry['name']==name].date[-1].year)
    if df_camp_mabry[df_camp_mabry['name']==name].date[0].year <=2000 and df_camp_mabry[df_camp_mabry['name']==name].date[-1].year>=2020:
        # Found that a number of stations were started in the year 2013 around Austin with continuous data
        stations_with_long_term_data.append(name)
print(stations_with_long_term_data)

df_stations = pd.DataFrame()
cols = ['value']
for name in stations_with_long_term_data:
    print(name)
    #df_camp_mabry[cols][df_camp_mabry['name']=='TX AUSTIN-CAMP MABRY'].plot()
    # df_camp_mabry[cols][df_camp_mabry['name']==name].plot()
    df_stations = pd.concat([df_stations, df_camp_mabry[df_camp_mabry['name']==name]])
    #df_stations.append(df_camp_mabry[df_camp_mabry['name']==name])
df_stations.to_csv('delhi_stations.csv', sep=',')

# This option stays to give the user a choice to select stations without nans
stations_without_nan = []
cols = ['value']
for name in df_camp_mabry.name.unique():
    #print(name)
    #df_camp_mabry[cols][df_camp_mabry['name']=='TX AUSTIN-CAMP MABRY'].plot()
    #print(df_camp_mabry[cols][df_camp_mabry['name']==name][:-15].isnull().values.any())
    if not df_camp_mabry[cols][df_camp_mabry['name']==name][:-15].isnull().values.any():
        stations_without_nan.append(name)
    #print(df_camp_mabry[cols][df_camp_mabry['name']==name])
print(stations_without_nan)
#sys.exit()
cond = (df_stations['date'].index.year<=2022) & (df_stations['date'].index.year>=2000)
#df_stations[cond]
print(df_stations[cond])
# Loop over the datetime for which all stations are available and then make supervised learning dataset for downscalebench

# df_stations_ = df_stations[cond]
# cond = df_stations.name == [stations_without_nan]
print(df_stations[df_stations['name'].isin(stations_with_long_term_data)].name.unique())
df_stations =  df_stations[df_stations['name'].isin(stations_with_long_term_data)]

if download_data:
    for year in range(1981,2024):
        print('wget https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.'+str(year)+'.days_p05.nc')
        cmd = 'wget https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.'+str(year)+'.days_p05.nc'
        os.system(cmd)

import xarray as xr
ds = xr.open_mfdataset('chirps-v2.0.*.days_p05.nc').sel(latitude=slice(delhi_lon_lat[1]-0.5, delhi_lon_lat[1]+0.5)).sel(longitude=slice(delhi_lon_lat[0]-0.5, delhi_lon_lat[0]+0.5))
print(ds)
print('writing delhi chirps netcdf file')
ds.to_netcdf('chirps_delhi_1981_2023.nc')
#!mv chirps_delhi_1981_2023.nc /content/drive/MyDrive/meteogan

#ds = xr.open_dataset('/content/drive/MyDrive/meteogan/chirps_delhi_1981_2023.nc')
ds = xr.open_dataset('chirps_delhi_1981_2023.nc')

df = ds.precip.mean(dim='time').to_dataframe().reset_index().dropna()
print(df)
#sys.exit()
#import plotly.express as px
## df = px.data.carshare()
#fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="precip",
#                  color_continuous_scale=px.colors.sequential.Viridis_r, size_max=8, zoom=7, size="precip")#,
#                  #mapbox_style="carto-positron")
#fig.update_layout(mapbox_style="open-street-map",autosize=True)
#fig.savefig('city_mapbox.pdf')
#fig.show()

#!pip install cartopy

#!pip install osmnx geopandas cartopy matplotlib

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#sys.exit()
df_stations_unique = df_stations.drop_duplicates(subset='name', keep='first')
#df_stations_ = df_stations[df_stations.name=='TX AUSTIN-CAMP MABRY']
df_stations_ = df_stations[df_stations.name=='NEW DELHI/SAFDARJUN']

import osmnx as ox
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
def plot_city_boundary_with_cartopy(city_name, country, ax):
    # Download city boundaries using osmnx
    city = ox.geocode_to_gdf(f"{city_name}, {country}")

    # Create a cartopy plot

    ax.set_title(f"Boundary of {city_name}")

    # Add natural features for context
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot the city boundary
    city.boundary.plot(ax=ax, color='red')

    # Set the extent to the city boundary with some margin
    margin = 0.1
    minx, miny, maxx, maxy = city.total_bounds
    ax.set_extent([minx - margin, maxx + margin, miny - margin, maxy + margin], crs=ccrs.PlateCarree())

    #plt.show()

# Plotting the boundary for Delhi, India
#plot_city_boundary_with_cartopy("Delhi", "India", ax=ax)
#ds.precip.mean(dim='time').plot(cmap='Blues', ax=ax)
#fig.savefig('austin_with_boundaries.png')
#for i in range(1,df_stations_unique.shape[0]):
#    print(df_stations_unique.lat.iloc[i], df_stations_unique.lon.iloc[i])
#
#    # Latitude and Longitude to mark
#    lat, lon, name = df_stations_unique.lat.iloc[i], df_stations_unique.lon.iloc[i], df_stations_unique.name.iloc[i]
#
#    # Mark the specified latitude and longitude
#    ax.plot(lon, lat, 'ro', markersize=5, transform=ccrs.PlateCarree())
#    ax.text(lon, lat, name, transform=ccrs.PlateCarree())
#
#df_stations[df_stations.name=='TX AUSTIN-CAMP MABRY'].value.plot()

if download_data:
    ds = xr.open_mfdataset('chirps-v2.0.*.days_p05.nc').sel(latitude=slice(delhi_lon_lat[1]-0.5, delhi_lon_lat[1]+0.5)).sel(longitude=slice(delhi_lon_lat[0]-0.5, delhi_lon_lat[0]+0.5)).compute()
else:
    ds = xr.open_dataset('chirps_delhi_1981_2023.nc')
ds_station = ds.precip
ds_station_gridded = ds_station.sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')

gridded = []
station = []
time = []

for i in range(ds_station_gridded.shape[0]):
    try:
        station.append(df_stations_[df_stations_.date==ds_station_gridded.time[i].values.astype('M8[D]').astype('O').strftime('%Y-%m-%d')].value[0])
        gridded.append(ds_station_gridded.values[i])
        #time.append(time[i].values.astype('M8[D]').astype('O').strftime('%Y-%m-%d'))
    except:
        continue

import numpy as np
df_compare = pd.DataFrame()
df_compare['gridded'] = np.array(gridded)
df_compare['station'] = np.array(station)
df_compare = df_compare.dropna()

print(df_compare)

import numpy as np

def rmse(predictions, targets):
    """
    Calculate the Root Mean Square Error (RMSE) between predictions and targets.

    Parameters:
    - predictions: Array-like, predicted values.
    - targets: Array-like, true values.

    Returns:
    - RMSE value.
    """
    differences = predictions - targets  # Subtract values
    differences_squared = differences ** 2  # Square the differences
    mean_difference_squared = differences_squared.mean()  # Find the mean of the squared differences
    rmse_val = np.sqrt(mean_difference_squared)  # Take the square root of the mean
    return rmse_val

# Example usage:
predictions = df_compare.gridded.values
targets = df_compare.station.values
print(rmse(predictions, targets))

ds = ds.precip.compute()

print(ds.values.shape)
#sys.exit()
from scipy.ndimage import zoom
print('zooming once')


reduced_data = zoom(ds.values, (1, 0.5, 0.5), order=3)

# Ensure the shape is exactly (15583, 10, 10)
reduced_data = reduced_data[:, :10, :10]

print('zooming 2x')
# Resize it back to the original size using bicubic interpolation
resized_data = zoom(reduced_data, (1, 2, 2), order=3)

# Ensure the shape is exactly (15583, 20, 20)
resized_data = resized_data[:, :20, :20]

resized_data_2x =  zoom(resized_data, (1, 2, 2), order=3) # 2.5 km
resized_data_4x =  zoom(resized_data_2x, (1, 2, 2), order=3) # 1.25 km
resized_data_8x =  zoom(resized_data_4x, (1, 2, 2), order=3) # 625 m
#resized_data_16x =  zoom(resized_data_8x, (1, 2, 2), order=3) # 312 m

# Define new 2x finer resolution latitude and longitude coordinates
new_lat = np.linspace(np.min(ds.latitude.values), np.max(ds.latitude.values), 2*len(ds.latitude.values) )
new_lon = np.linspace(np.min(ds.longitude.values), np.max(ds.longitude.values), 2*len(ds.longitude.values))
# Interpolate to the new coordinates
ds_interp_2x = ds.interp(latitude=new_lat, longitude=new_lon)

# Define new 2x finer resolution latitude and longitude coordinates
new_lat = np.linspace(np.min(ds_interp_2x.latitude.values), np.max(ds_interp_2x.latitude.values), 2*len(ds_interp_2x.latitude.values) )
new_lon = np.linspace(np.min(ds_interp_2x.longitude.values), np.max(ds_interp_2x.longitude.values), 2*len(ds_interp_2x.longitude.values))
# Interpolate to the new coordinates
ds_interp_4x = ds_interp_2x.interp(latitude=new_lat, longitude=new_lon)

# Define new 2x finer resolution latitude and longitude coordinates
new_lat = np.linspace(np.min(ds_interp_4x.latitude.values), np.max(ds_interp_4x.latitude.values), 2*len(ds_interp_4x.latitude.values) )
new_lon = np.linspace(np.min(ds_interp_4x.longitude.values), np.max(ds_interp_4x.longitude.values), 2*len(ds_interp_4x.longitude.values))
# Interpolate to the new coordinates
ds_interp_8x = ds_interp_4x.interp(latitude=new_lat, longitude=new_lon)

# # Define new 2x finer resolution latitude and longitude coordinates
# new_lat = np.linspace(np.min(ds_interp_8x.latitude.values), np.max(ds_interp_8x.latitude.values), 2*len(ds_interp_8x.latitude.values) - 1)
# new_lon = np.linspace(np.min(ds_interp_8x.longitude.values), np.max(ds_interp_8x.longitude.values), 2*len(ds_interp_8x.longitude.values) - 1)
# # Interpolate to the new coordinates
# ds_interp_16x = ds_interp_8x.interp(latitude=new_lat, longitude=new_lon)

ds['bicubic_5km'] = (('time', 'latitude', 'longitude'), resized_data)
ds_interp_2x['bicubic_2_5km'] = (('time', 'latitude', 'longitude'), resized_data_2x)
ds_interp_4x['bicubic_1_25km'] = (('time', 'latitude', 'longitude'), resized_data_4x)
ds_interp_8x['bicubic_600m'] = (('time', 'latitude', 'longitude'), resized_data_8x)
# ds_interp_16x['bicubic_300m'] = (('time', 'latitude', 'longitude'), resized_data_16x)
generator_target = ds.values
generator_input = ds.bicubic_5km.values
#discriminator_target = ds.stations.values


import osmnx as ox
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
def plot_city_boundary_with_cartopy(city_name, country, ax):
    # Download city boundaries using osmnx
    city = ox.geocode_to_gdf(f"{city_name}, {country}")

    # Create a cartopy plot

    ax.set_title(f"Boundary of {city_name}")

    # Add natural features for context
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot the city boundary
    city.boundary.plot(ax=ax, color='red')

    # Set the extent to the city boundary with some margin
    margin = 0.1
    minx, miny, maxx, maxy = city.total_bounds
    ax.set_extent([minx - margin, maxx + margin, miny - margin, maxy + margin], crs=ccrs.PlateCarree())

    #plt.show()

# Plotting the boundary for Delhi, India
#plot_city_boundary_with_cartopy("Delhi", "India", ax=ax)
#ds['bicubic_5km'].mean(dim='time').plot(cmap='Blues', ax=ax)
#sys.exit()
for i in range(1,df_stations_unique.shape[0]):
    print(df_stations_unique.lat.iloc[i], df_stations_unique.lon.iloc[i])

    # Latitude and Longitude to mark
    lat, lon, name = df_stations_unique.lat.iloc[i], df_stations_unique.lon.iloc[i], df_stations_unique.name.iloc[i]

    # Mark the specified latitude and longitude
    ax.plot(lon, lat, 'ro', markersize=5, transform=ccrs.PlateCarree())
    ax.text(lon, lat, name, transform=ccrs.PlateCarree())

ds_station = ds['bicubic_5km'].compute()
ds_station_gridded = ds_station.sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')
ds_station_gridded_2x = ds_interp_2x['bicubic_2_5km'].sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')
ds_station_gridded_4x = ds_interp_4x['bicubic_1_25km'].sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')
ds_station_gridded_8x = ds_interp_8x['bicubic_600m'].sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')

gridded = []
gridded_2x = []
gridded_4x = []
gridded_8x = []
station = []
time = []

for i in range(ds_station_gridded.shape[0]):
    try:
        station.append(df_stations_[df_stations_.date==ds_station_gridded.time[i].values.astype('M8[D]').astype('O').strftime('%Y-%m-%d')].value[0])
        gridded.append(ds_station_gridded.values[i])
        gridded_2x.append(ds_station_gridded_2x.values[i])
        gridded_4x.append(ds_station_gridded_4x.values[i])
        gridded_8x.append(ds_station_gridded_8x.values[i])
        #time.append(time[i].values.astype('M8[D]').astype('O').strftime('%Y-%m-%d'))
    except:
        continue

import numpy as np
df_compare = pd.DataFrame()
df_compare['gridded'] = np.array(gridded)
df_compare['gridded_2x'] = np.array(gridded_2x)
df_compare['gridded_4x'] = np.array(gridded_4x)
df_compare['gridded_8x'] = np.array(gridded_8x)
df_compare['station'] = np.array(station)



#df_compare = df_compare.dropna()
#predictions = df_compare.gridded.values
#targets = df_compare.station.values
#print('At 5 km, RMSE = ',rmse(predictions, targets))
#
#predictions = df_compare.gridded_2x.values
#targets = df_compare.station.values
#print('At 2.5 km, RMSE = ',rmse(predictions, targets))
#
#predictions = df_compare.gridded_4x.values
#targets = df_compare.station.values
#print('At 1.25 km, RMSE = ',rmse(predictions, targets))
#
#predictions = df_compare.gridded_8x.values
#targets = df_compare.station.values
#print('At 600 m, RMSE = ',rmse(predictions, targets))

import numpy as np

def psnr(original, compressed):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original and compressed images.

    Parameters:
    - original: Array-like, original image.
    - compressed: Array-like, compressed or reconstructed image.

    Returns:
    - PSNR value.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(original)
    psnr_val = 10 * np.log10(max_pixel**2 / mse)
    return psnr_val

# Example usage:
original = df_compare.station.values
compressed = df_compare.gridded.values
print('At 5 km, PSNR = ',psnr(original, compressed))  # Outputs: 24.130

original = df_compare.station.values
compressed = df_compare.gridded_2x.values
print('At 2.5 km, PSNR = ',psnr(original, compressed))  # Outputs: 24.130

original = df_compare.station.values
compressed = df_compare.gridded_4x.values
print('At 1.25 km, PSNR = ',psnr(original, compressed))  # Outputs: 24.130

original = df_compare.station.values
compressed = df_compare.gridded_8x.values
print('At 600 m, PSNR = ',psnr(original, compressed))  # Outputs: 24.130

df_stations_saf = df_stations[df_stations.name=='NEW DELHI/SAFDARJUN']
date_range_1 = pd.DatetimeIndex(ds.time.values)
date_range_2 = df_stations_saf.index
overlapping_dates = date_range_2.intersection(date_range_1)
# selected_rows = df_stations[(df_stations.name == 'NEW DELHI/SAFDARJUN') & (df_stations.index.isin(overlapping_dates))].sort_index()
selected_rows = df_stations_saf[df_stations_saf.index.isin(overlapping_dates)].sort_index()

#print('date_range_1 = ', date_range_1)
#print('date_range_2 = ', date_range_2)
#print('overlapping dates = ', overlapping_dates)
ds = ds.sel(time=overlapping_dates).copy()
# Create new dim/coordinate for stations
ds['stations'] = (('time'), selected_rows.value.values)

discriminator_target = ds.stations.values
#ds.stations.plot()

#print(generator_target.shape, generator_input.shape)
#print('before extracting time, ds_interp_2x = ', ds_interp_2x)
#print('ds_interp_2x.time = ', ds_interp_2x.time)
#print('ds.tiime.values = ', ds.time.values)
#print('overlapping dates = ', overlapping_dates)
ds_interp_2x = ds_interp_2x.sel(time=overlapping_dates).copy()
print('testing ds.shape, ds_interp_2x.shape ', ds.shape, ds_interp_2x.shape)


# SRCNN
# Input ds.bicubic_5km
# Target ds.precip

#!pip install tensorboard

#%load_ext tensorboard

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import os
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ncDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).unsqueeze(0)
        y = torch.from_numpy(self.targets[index]).unsqueeze(0)
        # x = self.data[index]
        # y = self.targets[index]
        # x = x.to(dtype=torch.float32)
        # y = y.to(dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.data)

x_train = generator_input.astype(np.float32)
y_train = generator_target.astype(np.float32)
print(generator_input.shape)


x_train_max = x_train.max()
y_train_max = y_train.max()
x_train /= x_train_max
y_train /= y_train_max

x_val_ = x_train[10000:12000]
y_val_ = y_train[10000:12000]

x_test_ = x_train[12000:]
y_test_ = y_train[12000:]

x_train_ = x_train[:10000]
y_train_ = y_train[:10000]

train_dataset = ncDataset(x_train_, y_train_)
val_dataset = ncDataset(x_val_, y_val_)
test_dataset = ncDataset(x_test_, y_test_)

lr, hr = train_dataset.__getitem__(0)
print(lr.shape, hr.shape, train_dataset.__len__())

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)

for batch in train_dataloader:
    data, targets = batch
    print(data.size())  # Should print torch.Size([20, 1, 30, 30])
    print(targets.size())  # Should print torch.Size([20, 1, 30, 601])
    break

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        lr, hr = batch
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    return train_loss, val_loss

# Initialize the model, loss function, and optimizer
device = 'cuda'
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#writer = SummaryWriter("runs/srcnn")

from copy import deepcopy

num_epochs = 1000
print_interval = 10
patience = 50
best_val_loss = float('inf')
counter = 0
best_model = None


for epoch in range(1, num_epochs + 1):
    train_loss, val_loss = train(model, train_dataloader, val_dataloader, criterion, optimizer, device)
# Log losses to TensorBoard
    #writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
    print("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = deepcopy(model)
        counter = 0
    else:
        counter += 1

    if epoch % print_interval == 0:
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if counter >= patience:
        print("Early stopping triggered.")
        break
#writer.close()

model_save_path = "best_model_srcnn.pth"
torch.save(best_model.state_dict(), model_save_path)

loaded_model = SRCNN().to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()

for batch in test_dataloader:
    lr, hr = batch
    lr, hr = lr.to(device), hr.to(device)
    with torch.no_grad():
        sr = loaded_model(lr)

generator_target = ds.values
generator_input = ds.bicubic_5km.values
x_train = generator_input.astype(np.float32)[:,np.newaxis,:,:]
y_train = generator_target.astype(np.float32)

x_train_max = x_train.max()
y_train_max = y_train.max()
x_train /= x_train_max
y_train /= y_train_max

# x_train_patches = patchify(x_train[:600,:600], patch_size)[:,np.newaxis,:,:]
x_train_tensor = torch.from_numpy(x_train).to(device)
with torch.no_grad():
    predicted_sr = loaded_model(x_train_tensor)
    predicted_sr_ = zoom(predicted_sr.cpu()[:,0,:,:], (1, 2, 2), order=3)

    predicted_sr_2x = loaded_model(torch.from_numpy(predicted_sr_[:,np.newaxis,:,:]).to(device))
    predicted_sr_2x_ = zoom(predicted_sr_2x.cpu()[:,0,:,:], (1, 2, 2), order=3)
    # torch.cuda.empty_cache() # move to A100 GPUs
    # predicted_sr_4x = loaded_model(torch.from_numpy(predicted_sr_2x_[:,np.newaxis,:,:]).to(device)) # move to A100 GPUs
    # predicted_sr_4x_ = zoom(predicted_sr_4x.cpu()[:,0,:,:], (1, 2, 2), order=3) # move to A100 GPUs

    # predicted_sr_8x = loaded_model(torch.from_numpy(predicted_sr_4x_[:,np.newaxis,:,:]).to(device)) # move to A100 GPUs
predicted_sr_np = predicted_sr.cpu().numpy() * y_train_max
predicted_sr_np_2x = predicted_sr_2x.cpu().numpy() * y_train_max
# predicted_sr_np_4x = predicted_sr_4x.cpu().numpy() * y_train_max # move to A100 GPUs
# predicted_sr_np_8x = predicted_sr_8x.cpu().numpy() * y_train_max # move to A100 GPUs

predicted_sr_np[predicted_sr_np<0] = 0.0
predicted_sr_np_2x[predicted_sr_np_2x<0] = 0.0
# predicted_sr_np_4x[predicted_sr_np_4x<0] = 0.0 # move to A100 GPUs
# predicted_sr_np_8x[predicted_sr_np_8x<0] = 0.0 # move to A100 GPUs

print(predicted_sr.shape, predicted_sr_2x.shape)

print(predicted_sr_np.shape, predicted_sr_np_2x.shape) #, predicted_sr_np_4x.shape, predicted_sr_np_8x.shape)

print(ds_interp_2x.shape, predicted_sr_2x.shape)


ds_interp_2x['srcnn_2_5km'] = (('time', 'latitude', 'longitude'), predicted_sr_np_2x[:,0,:,:])

ds['srcnn_5km'] = (('time', 'latitude', 'longitude'), predicted_sr_np[:,0,:,:])
# ds_interp_2x['srcnn_2_5km'] = (('time', 'latitude', 'longitude'), predicted_sr_np_2x[:,0,:,:])
#ds

import matplotlib.pyplot as plt
#fig,ax = plt.subplots(ncols=3,nrows=1, figsize=(15,5), subplot_kw={'projection': ccrs.PlateCarree()})
#ds.mean(dim='time').plot(cmap='magma_r', ax=ax[0], vmin=1.6, vmax=2.6, extend='both')
#ds.bicubic_5km.mean(dim='time').plot(cmap='magma_r', ax=ax[1], vmin=1.6, vmax=2.6, extend='both')
#ds.srcnn_5km.mean(dim='time').plot(cmap='magma_r', ax=ax[2], vmin=1.6, vmax=2.6, extend='both')
#titles = ['CHIRPS 5 km','Bicubic 5 km (PSNR =  26.690)', 'SRCNN 5 km (PSNR =  22.695)']
## for i_ax,ax in enumerate([ax[0], ax[1], ax[2]]):
##     plot_city_boundary_with_cartopy("Delhi", "India", ax=ax)
##     ax.set_title(titles[i_ax])
#plt.suptitle('Mean Precipitation over Delhi (1981-2023)')
#
#for i in range(1,df_stations_unique.shape[0]):
#    print(df_stations_unique.lat.iloc[i], df_stations_unique.lon.iloc[i])
#
#    # Latitude and Longitude to mark
#    lat, lon, name = df_stations_unique.lat.iloc[i], df_stations_unique.lon.iloc[i], df_stations_unique.name.iloc[i]
#
#    for i_ax,ax in enumerate([ax[0], ax[1], ax[2]]):
#        # Mark the specified latitude and longitude
#        ax.plot(lon, lat, 'ro', markersize=5, transform=ccrs.PlateCarree())
#        ax.text(lon, lat, name, transform=ccrs.PlateCarree())
#        plot_city_boundary_with_cartopy("Delhi", "India", ax=ax)
#        ax.set_title(titles[i_ax])
#
#plt.tight_layout()
#
#ds.stations.plot()
#
#import matplotlib.pyplot as plt
#fig,ax = plt.subplots(ncols=3,nrows=1, figsize=(15,5), subplot_kw={'projection': ccrs.PlateCarree()})
## ax = ax[0] = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
#
#ds.mean(dim='time').plot(cmap='magma_r', ax=ax[0], vmin=1.6, vmax=2.6, extend='both')
#ds.bicubic_5km.mean(dim='time').plot(cmap='magma_r', ax=ax[1], vmin=1.6, vmax=2.6, extend='both')
#ds.stations.plot(ax=ax[2])
#titles = ['CHIRPS 5 km (Generator Target)','Bicubic 5 km (Generator Input)', 'Station time seris']
## for i_ax,ax in enumerate([ax[0], ax[1], ax[2]]):
##     plot_city_boundary_with_cartopy("Delhi", "India", ax=ax)
##     ax.set_title(titles[i_ax])
#plt.suptitle('Supervised learning dataset using DownscaleBench')
#
#for i in range(1,df_stations_unique.shape[0]):
#    print(df_stations_unique.lat.iloc[i], df_stations_unique.lon.iloc[i])
#
#    # Latitude and Longitude to mark
#    lat, lon, name = df_stations_unique.lat.iloc[i], df_stations_unique.lon.iloc[i], df_stations_unique.name.iloc[i]
#
#    for i_ax,ax in enumerate([ax[0], ax[1]]):
#        # Mark the specified latitude and longitude
#        ax.plot(lon, lat, 'ro', markersize=5, transform=ccrs.PlateCarree())
#        ax.text(lon, lat, name, transform=ccrs.PlateCarree())
#        plot_city_boundary_with_cartopy("Delhi", "India", ax=ax)
#        ax.set_title(titles[i_ax])
#
#plt.tight_layout()
#
ds_station = ds['srcnn_5km'].compute()
ds_station_gridded = ds_station.sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')
## ds_station_gridded_2x = ds_interp_2x['bicubic_2_5km'].sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')
# ds_station_gridded_4x = ds_interp_4x['bicubic_1_25km'].sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')
# ds_station_gridded_8x = ds_interp_8x['bicubic_600m'].sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')

gridded = []
# gridded_2x = []
# gridded_4x = []
# gridded_8x = []
station = []
time = []

for i in range(ds_station_gridded.shape[0]):
    try:
        station.append(df_stations_[df_stations_.date==ds_station_gridded.time[i].values.astype('M8[D]').astype('O').strftime('%Y-%m-%d')].value[0])
        gridded.append(ds_station_gridded.values[i])
        # gridded_2x.append(ds_station_gridded_2x.values[i])
        # gridded_4x.append(ds_station_gridded_4x.values[i])
        # gridded_8x.append(ds_station_gridded_8x.values[i])
        #time.append(time[i].values.astype('M8[D]').astype('O').strftime('%Y-%m-%d'))
    except:
        continue

import numpy as np
df_compare = pd.DataFrame()
df_compare['gridded'] = np.array(gridded)
# df_compare['gridded_2x'] = np.array(gridded_2x)
# df_compare['gridded_4x'] = np.array(gridded_4x)
# df_compare['gridded_8x'] = np.array(gridded_8x)
df_compare['station'] = np.array(station)



df_compare = df_compare.dropna()
predictions = df_compare.gridded.values
targets = df_compare.station.values
print('At 5 km, RMSE = ',rmse(predictions, targets))
print('At 5 km, PSNR = ',psnr(predictions, targets))

# predictions = df_compare.gridded_2x.values
# targets = df_compare.station.values
# print('At 2.5 km, RMSE = ',rmse(predictions, targets))

# predictions = df_compare.gridded_4x.values
# targets = df_compare.station.values
# print('At 1.25 km, RMSE = ',rmse(predictions, targets))

# predictions = df_compare.gridded_8x.values
# targets = df_compare.station.values
# print('At 600 m, RMSE = ',rmse(predictions, targets))

import torch
from torch.utils.data import Dataset

class ncDataset(Dataset):
    def __init__(self, data, targets, discriminator_targets):
        self.data = data
        self.targets = targets
        self.discriminator_targets = discriminator_targets

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).unsqueeze(0).to(dtype=torch.float32)
        y = torch.from_numpy(self.targets[index]).unsqueeze(0).to(dtype=torch.float32)

        # Check if the discriminator target is a single number
        if isinstance(self.discriminator_targets[index], np.ndarray):
            d_y = torch.from_numpy(self.discriminator_targets[index]).unsqueeze(0).to(dtype=torch.float32)
        else:
            # If it's a single number, wrap it in a NumPy array
            d_y = torch.tensor([self.discriminator_targets[index]], dtype=torch.float32).unsqueeze(0)

        return x, y, d_y

    def __len__(self):
        return len(self.data)

ds_station_ = ds_station.sel(time=ds.time.values)
generator_input = ds_interp_2x
generator_target = ds_interp_2x.srcnn_2_5km
discriminator_target = ds_station.stations.sel(time=ds.time.values)

# For station loss .sel(longitude=df_stations_.lon.iloc[0], method='nearest').sel(latitude=df_stations_.lat.iloc[0], method='nearest')

#ds_station.stations.sel(time=ds.time.values).plot()

# Remove times for which station is nan
times_with_nan = discriminator_target.isnull()

# Step 2: Use these indices to drop the corresponding times from both data1 and data2
generator_input_ = generator_input.where(~times_with_nan, drop=True)
generator_target_ = generator_target.where(~times_with_nan, drop=True)
discriminator_target_ = discriminator_target.where(~times_with_nan, drop=True)

x_train = generator_input_.values.astype(np.float32)
y_train = generator_target_.values.astype(np.float32)
d_y = discriminator_target_.values.astype(np.float32)

x_train_max = x_train.max()
y_train_max = y_train.max()
d_y_train_max = d_y.max()

x_train /= x_train_max
y_train /= y_train_max
d_y_train =  d_y / d_y_train_max

x_val_ = x_train[6000:7000]
y_val_ = y_train[6000:7000]
d_y_val_ = d_y_train[6000:7000]

x_test_ = x_train[7000:]
y_test_ = y_train[7000:]
d_y_test_ = d_y_train[7000:]

x_train_ = x_train[:6000]
y_train_ = y_train[:6000]
d_y_train_ = d_y_train[:6000]

train_dataset = ncDataset(x_train_, y_train_, d_y_train_)
val_dataset = ncDataset(x_val_, y_val_, d_y_val_)
test_dataset = ncDataset(x_test_, y_test_, d_y_test_)

lr, hr, disc = train_dataset.__getitem__(0)
print(lr.shape, hr.shape, disc.shape, train_dataset.__len__())

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)

for batch in train_dataloader:
    data, targets, discs = batch
    print(data.size())  # Should print torch.Size([20, 1, 30, 30])
    print(targets.size())  # Should print torch.Size([20, 1, 30, 601])
    print(discs.size())  # Should print torch.Size([20, 1, 30, 601])
    break

print(generator_input_.shape, generator_target_.shape, discriminator_target_.shape)

#np.sum(np.isnan(discriminator_target_))

def find_nearest(data, target_lat, target_lon):
    # Calculate the absolute difference between target and all data points
    lat_diff = np.abs(data.latitude - target_lat)
    lon_diff = np.abs(data.longitude - target_lon)

    # Identify the indices of the minimum values
    min_lat = lat_diff.argmin()
    min_lon = lon_diff.argmin()

    return min_lat.item(), min_lon.item()  # Convert to Python int for usability

target_lat = df_stations_.lat.iloc[0]  # Example latitude
target_lon = df_stations_.lon.iloc[0]  # Example longitude
i, j = find_nearest(generator_input_, target_lat, target_lon)
station_coords = [i,j]
print(i,j)

import torch
import torch.nn as nn
import torch.nn.functional as F

class StationLoss(nn.Module):
    def __init__(self):
        super(StationLoss, self).__init__()

    def forward(self, output, ground_truth, station_coords):
        """
        Compute the loss based on station data.

        :param output: The output from the SRCNN model converted to xarray
        :param ground_truth: The ground truth values at the station locations - convert to xarray later
        :return: The computed loss.
        """
        # Extract values from the output at station locations
        #values_at_stations = output[:, station_coords[:, 0], station_coords[:, 1]]
        values_at_stations = output[:, 0, station_coords[0], station_coords[1]]

        # Compute the loss
        loss = F.mse_loss(values_at_stations, ground_truth)

        return loss

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, station_coords):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        lr, hr, disc = batch
        lr, hr, disc = lr.to(device), hr.to(device), disc.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        #print(sr.shape)
        loss = criterion(sr, disc, station_coords)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            # lr, hr = batch
            # lr, hr = lr.to(device), hr.to(device)
            lr, hr, disc = batch
            lr, hr, disc = lr.to(device), hr.to(device), disc.to(device)
            sr = model(lr)
            loss = criterion(sr, disc, station_coords)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    return train_loss, val_loss

# Initialize the model, loss function, and optimizer
device = 'cuda'
model = SRCNN().to(device)
criterion = StationLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#!pip install tqdm
from tqdm import tqdm

from copy import deepcopy

num_epochs = 10000
print_interval = 100
patience = 1000
best_val_loss = float('inf')
counter = 0
best_model = None


for epoch in tqdm(range(1, num_epochs + 1)):
    train_loss, val_loss = train(model, train_dataloader, val_dataloader, criterion, optimizer, device, station_coords)
# Log losses to TensorBoard
    #writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
    print("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = deepcopy(model)
        counter = 0
    else:
        counter += 1

    if epoch % print_interval == 0:
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if counter >= patience:
        print("Early stopping triggered.")
        break
#writer.close()
model_save_path = "best_model_srcnn_stationloss_delhi.pth"
torch.save(best_model.state_dict(), model_save_path)


# Generator
# Input ds.bicubic_5km
# Target ds.precip

# Discriminator
# Input: Output of Generator
# Target: Station

#import torch
#import torch.nn as nn
#from torch.autograd import Variable
#from torchvision.models import vgg19
#import torch.nn as nn
#
#class ResidualBlock(nn.Module):
#    def __init__(self, in_channels):
#        super(ResidualBlock, self).__init__()
#        self.block = nn.Sequential(
#            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(in_channels),
#            nn.PReLU(),
#            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(in_channels)
#        )
#
#    def forward(self, x):
#        return x + self.block(x)
#
#class UpsampleBlock(nn.Module):
#    def __init__(self, in_channels, scale_factor):
#        super(UpsampleBlock, self).__init__()
#        self.block = nn.Sequential(
#            nn.Conv2d(in_channels, in_channels * scale_factor**2, kernel_size=3, padding=1),
#            nn.PixelShuffle(scale_factor),
#            nn.PReLU()
#        )
#
#    def forward(self, x):
#        return self.block(x)
#
#
#class Generator(nn.Module):
#    def __init__(self, scale_factor):
#        upsample_block_num = int(scale_factor / 2)
#        super(Generator, self).__init__()
#
#        self.block1 = nn.Sequential(
#            nn.Conv2d(3, 64, kernel_size=9, padding=4),
#            nn.PReLU()
#        )
#        self.block2 = ResidualBlock(64)
#        self.block3 = nn.Sequential(
#            nn.Conv2d(64, 64, kernel_size=3, padding=1),
#            nn.BatchNorm2d(64)
#        )
#        block4 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
#        block4.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
#        self.block4 = nn.Sequential(*block4)
#
#    def forward(self, x):
#        block1 = self.block1(x)
#        block2 = self.block2(block1)
#        block3 = self.block3(block2)
#        block4 = self.block4(block1 + block3)
#        return (torch.tanh(block4) + 1) / 2
#
#
#
#class Discriminator(nn.Module):
#    def __init__(self):
#        super(Discriminator, self).__init__()
#        self.net = nn.Sequential(
#            nn.Conv2d(3, 64, kernel_size=3, padding=1),
#            nn.LeakyReLU(0.2, inplace=True),
#
#            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#            nn.BatchNorm2d(64),
#            nn.LeakyReLU(0.2, inplace=True),
#
#            nn.Conv2d(64, 128, kernel_size=3, padding=1),
#            nn.BatchNorm2d(128),
#            nn.LeakyReLU(0.2, inplace=True),
#
#            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
#            nn.BatchNorm2d(128),
#            nn.LeakyReLU(0.2, inplace=True),
#
#            nn.Conv2d(128, 256, kernel_size=3, padding=1),
#            nn.BatchNorm2d(256),
#            nn.LeakyReLU(0.2, inplace=True),
#
#            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
#            nn.BatchNorm2d(256),
#            nn.LeakyReLU(0.2, inplace=True),
#
#            nn.Conv2d(256, 512, kernel_size=3, padding=1),
#            nn.BatchNorm2d(512),
#            nn.LeakyReLU(0.2, inplace=True),
#
#            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
#            nn.BatchNorm2d(512),
#            nn.LeakyReLU(0.2, inplace=True),
#
#            nn.AdaptiveAvgPool2d(1),
#            nn.Conv2d(512, 1024, kernel_size=1),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Conv2d(1024, 1, kernel_size=1)
#        )
#
#    def forward(self, x):
#        return self.net(x)
#
#scale_factor = 4  # replace with the value you need
#generator = Generator(scale_factor)
#discriminator = Discriminator()
#
## Define Loss Functions
#criterion_G = nn.MSELoss()
#criterion_D = nn.BCEWithLogitsLoss()
#
## Define Optimizers
#optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
#optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
#
#import torch
#import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
#import numpy as np
#from PIL import Image
#import random
#
#class RandomImageDataset(Dataset):
#    def __init__(self, num_samples, hr_size, lr_size, transform=None):
#        self.num_samples = num_samples
#        self.hr_size = hr_size
#        self.lr_size = lr_size
#        self.transform = transform
#
#    def __len__(self):
#        return self.num_samples
#
#    def __getitem__(self, idx):
#        # Generating random images as numpy arrays
#        hr_image = np.random.rand(*self.hr_size, 3) * 255
#        lr_image = np.random.rand(*self.lr_size, 3) * 255
#
#        # Converting numpy arrays to PIL Images
#        hr_image = Image.fromarray(hr_image.astype('uint8'))
#        lr_image = Image.fromarray(lr_image.astype('uint8'))
#
#        if self.transform:
#            hr_image = self.transform(hr_image)
#            lr_image = self.transform(lr_image)
#
#        return hr_image, lr_image
#
#transform = transforms.Compose([
#    transforms.ToTensor(),
#])
#
#dataset = RandomImageDataset(num_samples=100, hr_size=(256, 256), lr_size=(64, 64), transform=transform)
#dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#
##!pip install tqdm
#
#num_epochs = 100
#for epoch in tqdm(range(num_epochs)):
#    for data in dataloader:
#        # Load data
#        high_res, low_res = data
#
#        # Train Discriminator
#        optimizer_D.zero_grad()
#        real_labels = torch.ones(high_res.size(0), 1)
#        fake_labels = torch.zeros(high_res.size(0), 1)
#
#        outputs = discriminator(high_res).view(-1, 1)
#        real_loss = criterion_D(outputs, real_labels)
#
#        fake_images = generator(low_res)
#        outputs = discriminator(fake_images.detach()).view(-1, 1)
#        fake_loss = criterion_D(outputs, fake_labels)
#
#        d_loss = real_loss + fake_loss
#        d_loss.backward()
#        optimizer_D.step()
#
#        # Train Generator
#        optimizer_G.zero_grad()
#        outputs = discriminator(fake_images).view(-1, 1)
#        g_adv_loss = criterion_D(outputs, real_labels)
#        g_content_loss = criterion_G(fake_images, high_res)
#        g_loss = g_adv_loss + 1e-3 * g_content_loss
#        g_loss.backward()
#        optimizer_G.step()
#
#    print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")
#
#torch.save(generator.state_dict(), 'generator.pth')
#torch.save(discriminator.state_dict(), 'discriminator.pth')
#
## For testing, load the model and run it on a low-resolution image
#generator.load_state_dict(torch.load('generator.pth'))
#low_res_test = ...  # Load a low-resolution image here
#high_res_fake = generator(low_res_test)
#
#
#
#
#
#def train(model, train_dataloader, val_dataloader, criterion, optimizer, device):
#    model.train()
#    train_loss = 0.0
#    for batch in train_dataloader:
#        lr, hr = batch
#        lr, hr = lr.to(device), hr.to(device)
#        optimizer.zero_grad()
#        sr = model(lr)
#
#        # output = model(dummy_input)  # You should replace dummy_input with your actual input data
#        # # Compute the loss
#        # loss = loss_function(output, dummy_ground_truth, dummy_station_coords)
#
#
#        loss = criterion(sr, hr)
#        loss.backward()
#        optimizer.step()
#        train_loss += loss.item()
#
#    train_loss /= len(train_dataloader)
#
#    # Validation
#    model.eval()
#    val_loss = 0.0
#    with torch.no_grad():
#        for batch in val_dataloader:
#            lr, hr = batch
#            lr, hr = lr.to(device), hr.to(device)
#            sr = model(lr)
#            loss = criterion(sr, hr)
#            val_loss += loss.item()
#
#    val_loss /= len(val_dataloader)
#
#    return train_loss, val_loss
