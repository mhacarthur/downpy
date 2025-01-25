import os
import json
import time
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

import cartopy.feature as cf
import cartopy.crs as ccrs

from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import psutil
from joblib import Parallel, delayed

import sys
sys.path.insert(0, os.path.abspath("../function"))
from ART_downscale import compute_beta
from ART_preprocessing import create_box, space_time_scales_agregations, wet_matrix_extrapolation

# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-pr", "--product", type=str, required=True)
parser.add_argument("-tr", "--time_reso", type=str, required=True)
parser.add_argument("-ys", "--yys", type=int, required=True)
parser.add_argument("-ye", "--yye", type=int, required=True)

args = vars(parser.parse_args())
product = vars(parser.parse_args())['product']
time_reso = vars(parser.parse_args())['time_reso']
yy_s = vars(parser.parse_args())['yys']
yy_e = vars(parser.parse_args())['yye']

lon_min, lon_max, lat_min, lat_max = 10.5, 13.5, 44.5, 47

# =============================================================================
json_read = f'../json/{product}_{time_reso}.json'
if os.path.exists(json_read):
    with open (json_read) as f:
        param = json.load(f)
else:
    raise SystemExit(f"File not found: {json_read}")

print(f'Read json file   : {json_read.split('/')[-1]}')
print(f'Number of threads: {param['BETA_cores']}')
print()

# =============================================================================
print(f'Reading data: {param['file']}')
dir_data = os.path.join(f'../data/{param['file']}')

PRE_data = xr.open_dataset(dir_data)
PRE_data = PRE_data.sel(time=PRE_data.time.dt.year.isin([np.arange(yy_s,yy_e+1)]))

if product == 'MSWEP':
    PRE_data = PRE_data.sel(lat=slice(lat_max+1.5, lat_min-1.5), lon=slice(lon_min-1.5, lon_max+1.5))
else:
    PRE_data = PRE_data.sel(lat=slice(lat_min-1.5, lat_max+1.5), lon=slice(lon_min-1.5, lon_max+1.5))

lats = PRE_data['lat'].data
lons = PRE_data['lon'].data

lon2d, lat2d = np.meshgrid(lons, lats)

nlon = np.size(lons)
nlat = np.size(lats)
ntime = len(PRE_data['time'])

year_vector = np.unique(pd.to_datetime(PRE_data['time']).year)

# =============================================================================
if product == 'MSWEP':
    ds_veneto = PRE_data.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
else:
    ds_veneto = PRE_data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

if time_reso == '3h':
    PRE_mean = ds_veneto.resample(time ='D').sum(dim='time', skipna=False).mean(dim='time', skipna=True)
elif time_reso == '1dy':
    PRE_mean = ds_veneto.mean(dim='time', skipna=True)
else:
    raise SystemExit(f"Time resolution not found: {time_reso}")

lat_ref = ds_veneto.lat.values
lon_ref = ds_veneto.lon.values

ndices_lat = np.where(np.isin(lats, lat_ref))[0]
ndices_lon = np.where(np.isin(lons, lon_ref))[0]

lon2d_ref, lat2d_ref = np.meshgrid(lon_ref, lat_ref)

del ds_veneto

# =============================================================================
def beta_3h_1dy(DATA_in, time_reso, lat_c, lon_c, param):
    if time_reso == '3h':
        PRE_daily = DATA_in.resample(time ='D').sum(dim='time', skipna=False)
        box_3h, _ = create_box(PRE_daily, lat_c, lon_c, param['npix'], param['radio'])
        tscales = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36, 48, 96])*param['dt']
    elif time_reso == '1dy':
        tscales = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*param['dt']
        box_3h, _ = create_box(DATA_in, lat_c, lon_c, param['npix'], param['radio'])
    else:
        print(f'Erorr: {time_reso} not valid')
        return None

    smax = box_3h.shape[0]
    tscales = tscales[tscales < param['tmax'] + 0.001]
    xscales = np.arange(1, smax+1)
    xscales_km = xscales*param['L1']

    WET_MATRIX = space_time_scales_agregations(
                box_3h, 
                param['L1'], 
                param['condition'], 
                tscales, 
                xscales, 
                param['npix'], 
                param['thresh'])

    nxscales = np.size(xscales)

    tscales_INTER = np.linspace(np.min(tscales), np.max(tscales), param['ninterp'])
    WET_MATRIX_INTER = np.zeros((param['ninterp'], nxscales))

    for col in range(nxscales):
        WET_MATRIX_INTER[:, col] = np.interp(tscales_INTER, tscales, WET_MATRIX[:, col])

    WET_MATRIX_EXTRA, new_spatial_scale = wet_matrix_extrapolation(
                WET_MATRIX_INTER, 
                xscales_km, 
                tscales_INTER, 
                param['L1'], 
                param['npix'])

    origin_ref = [param['origin_x'], param['origin_t']]
    target_ref = [param['target_x'], param['target_t']]

    beta = compute_beta(WET_MATRIX_EXTRA, origin_ref, target_ref, new_spatial_scale, tscales_INTER)

    return beta

# =============================================================================
def compute_for_point(lat_idx, lon_idx):
    return beta_3h_1dy(PRE_data, time_reso, lats[lat_idx], lons[lon_idx], param)

Resource = []

start_time = time.time()

results = Parallel(n_jobs=param['BETA_cores'])(
    delayed(compute_for_point)(la, lo) for la in ndices_lat for lo in ndices_lon
    )

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
memory_consumed = psutil.virtual_memory().used / 1024 ** 3

print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
print(f"Memory consumed: {memory_consumed:.3f} GB")
print()

# =============================================================================
if product == 'IMERG' and time_reso == '3h':
    Resource = pd.DataFrame({
                    'Product':product,
                    'Resolution_t':[time_reso],
                    'Cores':param['BETA_cores'],
                    'Time(min)':np.round(elapsed_minutes,3),
                    'memory':np.round(memory_consumed,3)})

    Resource.to_csv(f'../resources/VENETO_{product}_mean_beta_{time_reso}_cores_{str(param['BETA_cores']).zfill(2)}.csv', header=True, index=False)

# =============================================================================
BETA_VENETO = np.array(results).reshape(len(ndices_lat), len(ndices_lon))

BETA_xr = xr.Dataset(data_vars={"BETA": (("lat","lon"), BETA_VENETO.data)},
                    coords={'lat': lats[ndices_lat], 'lon': lons[ndices_lon]},
                    attrs=dict(description=f"Beta of {product} for Veneto region limited as 10.5E to 13.5E and 44.5N to 47N"))

BETA_xr.BETA.attrs["units"] = "dimensionless"
BETA_xr.BETA.attrs["long_name"] = "Relation between Origin and Tarjet wet fraction"
BETA_xr.BETA.attrs["origname"] = "BETA"

BETA_xr.lat.attrs["units"] = "degrees_north"
BETA_xr.lat.attrs["long_name"] = "Latitude"

BETA_xr.lon.attrs["units"] = "degrees_east"
BETA_xr.lon.attrs["long_name"] = "Longitude"

BETA_out = os.path.join('..','output',f'VENETO_BETA_{product}_{time_reso}_{yy_s}_{yy_e}.nc')
print(f'Export PRE data to {BETA_out}')
BETA_xr.to_netcdf(BETA_out)

print(f'BETA data saved in {BETA_out}')