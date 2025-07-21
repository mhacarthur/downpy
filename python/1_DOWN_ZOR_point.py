import os
import json
import time
import psutil
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath(".."))
from function import DOWN_raw

# =============================================================================
import itertools
from multiprocessing import Pool, cpu_count

# =============================================================================
# This Script export as csv the downscaled precipitation data for each point

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
full_years = np.arange(yy_s, yy_e + 1)

# =============================================================================
Tr = np.array([5, 10, 20, 50, 100])

# # Coast area 
# lon_min, lon_max, lat_min, lat_max, area, toll = 12, 12.5, 45.2, 45.7, 'COAST', 0.05
# # Fast area
# lon_min, lon_max, lat_min, lat_max, area, toll = 11.5, 12, 45.5, 46, 'FAST', 0.05
# # Test area
# lon_min, lon_max, lat_min, lat_max, area, toll = 11, 12.5, 45, 46.5, 'TEST', 0.05
# # Veneto area
# lon_min, lon_max, lat_min, lat_max, area, toll = 10.5, 13.5, 44.5, 47, 'VENETO', 0.002
# # Italy
lon_min, lon_max, lat_min, lat_max, area, toll = 6.5, 19, 36.5, 48, 'ITALY', 0.002

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

print(f'Json file   : {json_read.split('/')[-1]}')
print(f'Region      : {area}')
print(f'ACF func    : {param['acf']}')
print(f'Threshold   : {thresh}')
print(f'Threads     : {nproces}')
print(f'Neighborhood: {NEIBHR}x{NEIBHR}')
print()

# =============================================================================
print(f'Reading data: {param['file']}')
print()
dir_data_1 = os.path.join(f'../data/{param["file"]}')
dir_base = os.path.join('/', 'media', 'arturo', 'T9', 'Data', 'Italy', 'Satellite')
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

if product == 'MSWEP' or product == 'PERSIANN' or product == 'SM2RAIN' or product == 'ERA5' or product == 'GSMaP':
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
print(f'Extracting lat and lon points for area')
print()
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
def downscale_clear(DATA_3h,la,lo, param):

    lat_c = lat_ref[la]
    lon_c = lon_ref[lo]

    Tr = np.array([5, 10, 20, 50, 100, 200])

    box_3h = DOWN_raw.create_box_v2(DATA_3h, lat_c, lon_c, param['npix'])

    all_nan_mask = box_3h.isnull().all(dim='time')
    num_all_nan_pixels = all_nan_mask.sum().item()

    if num_all_nan_pixels == 0:
        print(f'Point: {la,lo}')
        downres = DOWN_raw.downscale(box_3h, Tr, thresh=1.0, L0=0.0001,
                                    cor_method=param['corr_method'], toll=toll,
                                    acf=param['acf'], save_yearly=True,
                                    maxmiss=40, clat=lat_c, clon=lon_c,
                                    opt_method=param['opt_method'], plot=False)

        downres_out = os.path.join('..','point',f'P_{la}_{lo}_{area}_{product}_{time_reso}.json')
        with open(downres_out, 'w') as f:
            json.dump(downres, f, default=str)

    else:
        downres = {}

    return downres

# =============================================================================
PRE_data_T = PRE_data.transpose('lon', 'lat', 'time')
# x = da.from_array(PRE_data_T['PRE'], chunks=(6, 6, 300))
time_vector_dt = pd.to_datetime(PRE_data_T['PRE']['time'].values)
DATA_3h = xr.DataArray(PRE_data_T['PRE'],  
                        coords={
                            'lon':PRE_data_T['lon'].values, 
                            'lat':PRE_data_T['lat'].values, 
                            'time':time_vector_dt},
                        dims=('lon', 'lat', 'time'))

# =============================================================================
# Test values
# la_indices = range(2)
# lo_indices = range(2)

# Full values
la_indices = range(len(lat_ref))
lo_indices = range(len(lon_ref))

point_list = list(itertools.product(la_indices, lo_indices))

def downscale_wrapper(args):
    la, lo, DATA_3h, param = args
    try:
        return downscale_clear(DATA_3h, la, lo, param)
    except Exception as e:
        print(f"Error at point ({la}, {lo}): {e}")
        return None

input_list = [(la, lo, DATA_3h, param) for (la, lo) in point_list]

print(f'Total points to process : {len(input_list)}')

filtered_input_list = []
for ll in input_list:
    verify_name = f'P_{ll[0]}_{ll[1]}_{area}_{product}_{time_reso}.json'
    file_path = os.path.join('..', 'point', verify_name)
    if not os.path.exists(file_path):
        filtered_input_list.append(ll)
input_list = filtered_input_list

print(f'Total points after clean: {len(input_list)}')
print()

if __name__ == '__main__':
    with Pool(processes=15) as pool:
        results = pool.map(downscale_wrapper, input_list)
