import os
import json
# import time
import argparse
import numpy as np
import xarray as xr
import pandas as pd
# import seaborn as sns
# import geopandas as gpd

# import dask.array as da
# from datetime import datetime

# import cartopy.crs as ccrs

# from matplotlib import patches
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from playsound import playsound
from multiprocessing import Pool

from scipy.optimize import differential_evolution

# import psutil
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath(".."))
# from function import ART_downscale as ART_down
# from function import ART_preprocessing as ART_pre
from function import DOWN_raw

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

npix = 3 #3x3
# npix = 4 # IMERG 5x5

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
elif product == 'PERSIANN':
    buffer = 0.50*npix*0.25
else:
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
NEIBHR = 2*param['npix']+1

print(f'Read json file   : {json_read.split('/')[-1]}')
print(f'ACF function     : {param['acf']}')
print(f'Number of threads: {param['BETA_cores']}')
print(f'Neighborhood area: {NEIBHR}x{NEIBHR}')
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
    PRE_veneto = PRE_data.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
else:
    PRE_veneto = PRE_data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

lat_ref = PRE_veneto.lat.values
lon_ref = PRE_veneto.lon.values

ndices_lat = np.where(np.isin(lats, lat_ref))[0]
ndices_lon = np.where(np.isin(lons, lon_ref))[0]

lon2d_ref, lat2d_ref = np.meshgrid(lon_ref, lat_ref)

del PRE_veneto

# =============================================================================
def downscale_clear(DATA_RDY,la,lo, buffer):
    
    lat_c = lat_ref[la]
    lon_c = lon_ref[lo]

    Tr = np.array([5, 10, 20, 50, 100, 200])

    # eps = 1e-4
    # solat = lat_c - buffer + eps
    # nolat = lat_c + buffer + eps
    # ealon = lon_c + buffer + eps
    # welon = lon_c - buffer + eps
    # bcond = np.logical_and(
    #             np.logical_and( DATA_RDY.lat > solat, DATA_RDY.lat < nolat),
    #             np.logical_and( DATA_RDY.lon > welon, DATA_RDY.lon < ealon))
    # box_3h = DATA_RDY.where(bcond, drop = True).load()

    box_3h = DOWN_raw.create_box_v2(DATA_RDY, lat_c, lon_c, param['npix'])
    
    # if box_3h.shape[0] !=3 or box_3h.shape[1] !=3:
    # if box_3h.shape[0] !=5 or box_3h.shape[1] !=5:
    #     print(lat_c,lon_c,box_3h.shape)

    downres = DOWN_raw.downscale(box_3h, Tr, thresh=1.0, L0=0.0001, 
                                cor_method=param["corr_method"], toll=0.005,
                                acf=param['acf'], save_yearly=True,
                                maxmiss=40, clat=lat_c, clon=lon_c,
                                opt_method='genetic', plot=False)

    return downres

# =============================================================================
PRE_data_T = PRE_data.transpose('lon', 'lat', 'time')
# x = da.from_array(PRE_data_T['PRE'], chunks=(6, 6, 300))
time_vector_dt = pd.to_datetime(PRE_data_T['PRE']['time'].values)
DATA_RDY = xr.DataArray(PRE_data_T['PRE'],  coords={
                            'lon':PRE_data_T['lon'].values, 
                            'lat':PRE_data_T['lat'].values, 
                            'time':time_vector_dt},
                            dims=('lon', 'lat', 'time'))

# =============================================================================
print(f'ACF: {param['acf']}')

def compute_for_point(args):
    DATA_RDY, la, lo, buffer = args
    return la, lo, downscale_clear(DATA_RDY,la,lo,buffer)

with Pool(processes=27) as pool:
    results = pool.map(compute_for_point, [(DATA_RDY,la,lo,buffer) for la in range(len(lat_ref)) for lo in range(len(lon_ref))])

# ==============================================================================
Tr = np.array([5, 10, 20, 50, 100, 200])

years_num = yy_e - yy_s + 1
full_years = np.arange(yy_s, yy_e + 1)

shape = (len(lat_ref), len(lon_ref))

NYs = np.full([years_num, len(lat_ref), len(lon_ref)], np.nan)
CYs = np.full([years_num, len(lat_ref), len(lon_ref)], np.nan)
WYs = np.full([years_num, len(lat_ref), len(lon_ref)], np.nan)

NYd = np.full([years_num, len(lat_ref), len(lon_ref)], np.nan)
CYd = np.full([years_num, len(lat_ref), len(lon_ref)], np.nan)
WYd = np.full([years_num, len(lat_ref), len(lon_ref)], np.nan)

Mev_d = np.zeros((len(Tr), *shape))
Mev_s = np.zeros((len(Tr), *shape))

BETA = np.zeros([len(lat_ref), len(lon_ref)])
GAMMA = np.zeros([len(lat_ref), len(lon_ref)])

for la, lo, downres in results:
    
    available_years = downres['YEARS'].astype(int) 
    indices = np.searchsorted(full_years, available_years)
    
    NYs[indices, la, lo] = downres['NYs']
    CYs[indices, la, lo] = downres['CYs']
    WYs[indices, la, lo] = downres['WYs']
    
    NYd[indices, la, lo] = downres['NYd']
    CYd[indices, la, lo] = downres['CYd']
    WYd[indices, la, lo] = downres['WYd']
    
    Mev_d[:, la, lo] = downres['mev_d']
    Mev_s[:, la, lo] = downres['mev_s']
    
    BETA[la, lo] = downres['beta']
    GAMMA[la, lo] = downres['gam_d']

# =============================================================================
DOWN_xr = xr.Dataset(data_vars={
                    "NYs": (("year","lat","lon"), NYs),
                    "CYs": (("year","lat","lon"), CYs),
                    "WYs": (("year","lat","lon"), WYs),
                    "Mev_s": (("Tr","lat","lon"), Mev_s),
                    "NYd": (("year","lat","lon"), NYd),
                    "CYd": (("year","lat","lon"), CYd),
                    "WYd": (("year","lat","lon"), WYd),
                    "Mev_d": (("Tr","lat","lon"), Mev_d),
                    "BETA": (("lat","lon"), BETA),
                    "GAMMA": (("lat","lon"), GAMMA)
                    },
    coords={'Tr':Tr,'lat': lat_ref, 'lon': lon_ref},
    attrs=dict(description=f"Downscaling for {product} in the region bounded by longitudes {lon_min} to {lon_max} and latitudes {lat_min} to {lat_max}, using '{param['acf']}' as the acf function, '{param['thresh']} mm' threshold and box size '{npix}x{npix}'."))

DOWN_xr.NYs.attrs["units"] = "day"
DOWN_xr.NYs.attrs["long_name"] = "Number of wet days"
DOWN_xr.NYs.attrs["origname"] = "wet days"

DOWN_xr.CYs.attrs["units"] = "dimensionless"
DOWN_xr.CYs.attrs["long_name"] = "Weibull scale parameter"
DOWN_xr.CYs.attrs["origname"] = "scale"

DOWN_xr.WYs.attrs["units"] = "dimensionless"
DOWN_xr.WYs.attrs["long_name"] = "Weibull shape parameter"
DOWN_xr.WYs.attrs["origname"] = "shape"

DOWN_xr.Mev_s.attrs["units"] = "mm/day"
DOWN_xr.Mev_s.attrs["long_name"] = "Satellite Maximum Quantiles"
DOWN_xr.Mev_s.attrs["origname"] = "Satellite quantiles"

DOWN_xr.NYd.attrs["units"] = "day"
DOWN_xr.NYd.attrs["long_name"] = "Downscale Number of wet days"
DOWN_xr.NYd.attrs["origname"] = "down wet days"

DOWN_xr.CYd.attrs["units"] = "dimensionless"
DOWN_xr.CYd.attrs["long_name"] = "Downscale Weibull scale parameter"
DOWN_xr.CYd.attrs["origname"] = "down scale"

DOWN_xr.WYd.attrs["units"] = "dimensionless"
DOWN_xr.WYd.attrs["long_name"] = "Downscale Weibull shape parameter"
DOWN_xr.WYd.attrs["origname"] = "down shape"

DOWN_xr.Mev_d.attrs["units"] = "mm/day"
DOWN_xr.Mev_d.attrs["long_name"] = "Downscaling Maximum Quantiles"
DOWN_xr.Mev_d.attrs["origname"] = "Downscaling quantiles"

DOWN_xr.lat.attrs["units"] = "degrees_north"
DOWN_xr.lat.attrs["long_name"] = "Latitude"

DOWN_xr.lon.attrs["units"] = "degrees_east"
DOWN_xr.lon.attrs["long_name"] = "Longitude"

# ==============================================================================
DOWN_out = os.path.join('..','output',f'VENETO_DOWN_{product}_{time_reso}_{yy_s}_{yy_e}_npix_{param['npix']}_thr_{param['thresh']}_acf_{param['acf']}_ZOR_{cor_method}.nc')
print(f'Export PRE data to {DOWN_out}')
DOWN_xr.to_netcdf(DOWN_out)