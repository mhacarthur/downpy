import os
import json
import time
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da
import geopandas as gpd

import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath(".."))
from function import DOWN_raw

# Now is working only for 3 pixels
npix = 3

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
years_num = yy_e - yy_s + 1

Tr = np.array([5, 10, 20, 50, 100])

if product == 'CMORPH':
    buffer = 0.50*npix*0.25
elif product == 'ERA5':
    buffer = 0.50*npix*0.25
elif product == 'IMERG':
    buffer = 0.25*npix*0.25
elif product == 'MSWEP':
    buffer = 0.25*npix*0.25
elif product == 'GSMaP':
    buffer = 0.25*npix*0.25

# Test area
lon_min, lon_max, lat_min, lat_max = 11, 11.5, 46, 46.5
# Veneto
# lon_min, lon_max, lat_min, lat_max = 10.5, 13.5, 44.5, 47

# =============================================================================
json_read = f'../json/{product}_{time_reso}.json'
if os.path.exists(json_read):
    with open (json_read) as f:
        param = json.load(f)
else:
    raise SystemExit(f"File not found: {json_read}")

nproces = param['BETA_cores']
thresh = param['thresh']
acf_fun = param['acf']

print(f'Read json file   : {json_read.split('/')[-1]}')
print(f'Number of threads: {nproces}')
print()

# =============================================================================
print(f'Reading data: {param['file']}')
print()
dir_data_1 = os.path.join(f'../data/{param["file"]}')
dir_base = os.path.join('/', 'media', 'arturo', 'Arturo', 'Data', 'Italy', 'Satellite')
if product == 'SM2RAIN':
    dir_data_2 = os.path.join(dir_base,product,'ASCAT',time_reso,param['file'])
else:
    dir_data_2 = os.path.join(dir_base,product,time_reso,param['file'])

if os.path.exists(dir_data_1):
    dir_data = dir_data_1
else:
    if os.path.exists(dir_data_2):
        dir_data = dir_data_2
    else:
        raise FileNotFoundError("Directory doesn't exist")

# =============================================================================
PRE_data = xr.open_dataset(dir_data)
PRE_data = PRE_data.sel(time=PRE_data.time.dt.year.isin([np.arange(yy_s,yy_e+1)]))

if product == 'MSWEP' or product == 'PERSIANN' or product == 'SM2RAIN' or product == 'ERA5':
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

PRE_data = PRE_data.where(PRE_data >= 0)  # Reemplaza valores negativos con NaN

# =============================================================================
# Extracting lat and lon points for Study area (VENETO)
if product == 'MSWEP' or product == 'PERSIANN' or product == 'SM2RAIN' or product == 'ERA5' or product == 'GSMaP':
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
def downscale_clear(DATA_INPUT,la,lo,Tr,thresh,acf_fun, buffer):

    lat_c = lat_ref[la]
    lon_c = lon_ref[lo]

    eps = 1e-4

    solat = lat_c - buffer + eps
    nolat = lat_c + buffer + eps
    ealon = lon_c + buffer + eps
    welon = lon_c - buffer + eps
    bcond = np.logical_and(
                np.logical_and( DATA_INPUT.lat > solat, DATA_INPUT.lat < nolat),
                np.logical_and( DATA_INPUT.lon > welon, DATA_INPUT.lon < ealon))

    box_3h = DATA_INPUT.where(bcond, drop = True).load()

    downres = DOWN_raw.downscale(box_3h, Tr, thresh=thresh, L0=0, toll=0.005,
                        acf=acf_fun, save_yearly=True,
                        maxmiss=36, clat=lat_c, clon=lon_c,
                        opt_method='genetic', plot=False)

    return downres

# =============================================================================
PRE_data_T = PRE_data.transpose('lon', 'lat', 'time')
x = da.from_array(PRE_data_T['PRE'], chunks=(6, 6, 300))
time_vector = PRE_data_T['PRE']['time'].values
time_vector_dt = pd.to_datetime(time_vector)
DATA_RDY = xr.DataArray(x,  coords={
                            'lon':PRE_data_T['lon'].values, 
                            'lat':PRE_data_T['lat'].values, 
                            'time':time_vector_dt},
                            dims=('lon', 'lat', 'time'))

# =============================================================================
shape = (len(lat_ref), len(lon_ref))

Mev_d = np.zeros((len(Tr), *shape))
Mev_s = np.zeros((len(Tr), *shape))

BETA = np.zeros(shape)
GAMMAd = np.zeros(shape)
GAMMAs = np.zeros(shape)

years_shape = (years_num, *shape)
NYs = np.zeros(years_shape)
CYs = np.zeros(years_shape)
WYs = np.zeros(years_shape)
NYd = np.zeros(years_shape)
CYd = np.zeros(years_shape)
WYd = np.zeros(years_shape)

Ns = np.zeros(shape)
Cs = np.zeros(shape)
Ws = np.zeros(shape)
Nd = np.zeros(shape)
Cd = np.zeros(shape)
Wd = np.zeros(shape)

# =============================================================================
def compute_for_point(args):
    la, lo, DATA_RDY, Tr, thresh, acf_fun, buffer = args
    return la, lo, downscale_clear(DATA_RDY, la, lo, Tr, thresh, acf_fun, buffer)

with Pool(processes=nproces) as pool:
    results = pool.map(compute_for_point, [(la, lo, DATA_RDY, Tr, thresh, acf_fun, buffer) for la in range(len(lat_ref)) for lo in range(len(lon_ref))])

# =============================================================================
for la, lo, downres in results:
    Mev_d[:, la, lo] = downres['mev_d']
    Mev_s[:, la, lo] = downres['mev_s']

    NYs[:, la, lo] = downres['NYs']
    CYs[:, la, lo] = downres['CYs']
    WYs[:, la, lo] = downres['WYs']

    NYd[:, la, lo] = downres['NYd']
    CYd[:, la, lo] = downres['CYd']
    WYd[:, la, lo] = downres['WYd']

    BETA[la, lo] = downres['beta']
    GAMMAd[la, lo] = downres['gam_d']
    GAMMAs[la, lo] = downres['gam_s']

    Ns[la, lo] = int(downres['Ns'])
    Cs[la, lo] = float(downres['Cs'])
    Ws[la, lo] = float(downres['Ws'])

    Nd[la, lo] = int(downres['Nd'])
    Cd[la, lo] = float(downres['Cd'])
    Wd[la, lo] = float(downres['Wd'])

# =============================================================================
DOWN_xr = xr.Dataset(data_vars={
                    "Ns": (("lat","lon"), Ns),
                    "Cs": (("lat","lon"), Cs),
                    "Ws": (("lat","lon"), Ws),
                    
                    "Nd": (("lat","lon"), Nd),
                    "Cd": (("lat","lon"), Cd),
                    "Wd": (("lat","lon"), Wd),
                    
                    "NYs": (("year","lat","lon"), NYs),
                    "CYs": (("year","lat","lon"), CYs),
                    "WYs": (("year","lat","lon"), WYs),
                    
                    "NYd": (("year","lat","lon"), NYd),
                    "CYd": (("year","lat","lon"), CYd),
                    "WYd": (("year","lat","lon"), WYd),
                    
                    "BETA": (("lat","lon"), BETA),
                    "GAMMAd": (("lat","lon"), GAMMAd),
                    "GAMMAs": (("lat","lon"), GAMMAs),
                    
                    "Mev_d": (("Tr","lat","lon"), Mev_d),
                    "Mev_s": (("Tr","lat","lon"), Mev_s)
                    },
                    coords={'year':year_vector,'Tr':Tr,'lat': lat_ref, 'lon': lon_ref},
                    attrs=dict(description=f"Downscaling for {product} in the region bounded by longitudes {lon_min} to {lon_max} and latitudes {lat_min} to {lat_max}, using '{acf_fun}' as the acf function, '{thresh} mm' threshold and box size '{npix}x{npix}'."))

# =============================================================================
DOWN_out = os.path.join('..','output',f'VENETO_DOWN_{product}_{time_reso}_{yy_s}_{yy_e}_npix_{npix}_thr_{thresh}_acf_{acf_fun}.nc')
print(f'Export PRE data to {DOWN_out}')
DOWN_xr.to_netcdf(DOWN_out)
