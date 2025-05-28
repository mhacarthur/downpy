import os
import json
import time
import psutil
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution

from shapely.geometry import Point
from scipy.ndimage import binary_dilation

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath(".."))
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
full_years = np.arange(yy_s, yy_e + 1)

Tr = np.array([5, 10, 20, 50, 100])
toll = 0.05

# =============================================================================
# # All Italy using shapefile and extending the area
area = 'Italy-SHP'

# GEOMETRY = gpd.read_file(os.path.join('..','geometry','Veneto.geojson'))
GEOMETRY = gpd.read_file(os.path.join('..','geometry','Valle_daosta.geojson'))

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

print(f'Region      : {area}')
print(f'Json file   : {json_read.split('/')[-1]}')
print(f'ACF func    : {param['acf']}')
print(f'Threshold   : {thresh}')
print(f'Threads     : {nproces}')
print(f'Neighborhood: {NEIBHR}x{NEIBHR}')
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

lats = PRE_data['lat'].data
lons = PRE_data['lon'].data

lon2d, lat2d = np.meshgrid(lons, lats)

nlon = np.size(lons)
nlat = np.size(lats)
ntime = len(PRE_data['time'])

year_vector = np.unique(pd.to_datetime(PRE_data['time']).year)

PRE_data = PRE_data.where(PRE_data >= 0)  # Reemplaza valores negativos con NaN

# =============================================================================
expand = 2

lon_flat = lon2d.ravel()
lat_flat = lat2d.ravel()

points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)], crs="EPSG:4326")
mask = points.within(GEOMETRY.union_all())

valid_lons = lon_flat[mask]
valid_lats = lat_flat[mask]

mask_2d = mask.values.reshape(lon2d.shape)

mask_expanded = binary_dilation(mask_2d, iterations=expand)
data_masked = PRE_data['PRE'].where(mask_expanded) 

i_valid, j_valid = np.where(mask_expanded)

lat_valid = lat2d[i_valid, j_valid]
lon_valid = lon2d[i_valid, j_valid]

df_points = pd.DataFrame({
    'ndices_lat': i_valid,
    'ndices_lon': j_valid,
    'lat_ref': lat_valid,
    'lon_ref': lon_valid
})

print(f'Number of valid points: {len(df_points)}')
print()

# =============================================================================
def downscale_clear(DATA_3h,la,lo, param):
    
    lat_c = lats[la]
    lon_c = lons[lo]

    Tr = np.array([5, 10, 20, 50, 100, 200])

    box_3h = DOWN_raw.create_box_v2(DATA_3h, lat_c, lon_c, param['npix'])

    all_nan_mask = box_3h.isnull().all(dim='time')
    num_all_nan_pixels = all_nan_mask.sum().item()

    if num_all_nan_pixels == 0:
        downres = DOWN_raw.downscale(box_3h, Tr, thresh=1.0, L0=0.0001, 
                                    cor_method=param['corr_method'], toll=toll,
                                    acf=param['acf'], save_yearly=True,
                                    maxmiss=40, clat=lat_c, clon=lon_c,
                                    opt_method=param['opt_method'], plot=False)
    else:
        downres = {}

    return downres

# =============================================================================
PRE_data_T = PRE_data.transpose('lon', 'lat', 'time')
time_vector_dt = pd.to_datetime(PRE_data_T['PRE']['time'].values)
DATA_3h = xr.DataArray(PRE_data_T['PRE'],  
                        coords={
                            'lon':PRE_data_T['lon'].values, 
                            'lat':PRE_data_T['lat'].values, 
                            'time':time_vector_dt},
                        dims=('lon', 'lat', 'time'))

# =============================================================================
print(f'Start with downscale processes')
print()
start_time = time.time()

def compute_for_point(args):
    DATA_3h, la, lo, param = args
    return la, lo, downscale_clear(DATA_3h,la,lo,param)

with Pool(processes=param['BETA_cores']) as pool:
    results = pool.map(compute_for_point, [(DATA_3h, la, lo, param) for la, lo in zip(df_points['ndices_lat'], df_points['ndices_lon'])])

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
memory_consumed = psutil.virtual_memory().used / 1024 ** 3

# =============================================================================
print(f'Export Downscale info to csv')
print()
INFO = pd.DataFrame({
                    'Product':product,
                    'Resolution_t':[time_reso],
                    'Ntimes':[ntime],
                    'Region':[area],
                    'Neighborhood':[NEIBHR],
                    'Cores':param['BETA_cores'],
                    'Time(min)':np.round(elapsed_minutes,3),
                    'Memory(Gb)':np.round(memory_consumed,3)
                    })

INFO.to_csv('../csv/DOWN_INFO.csv', 
            mode='a', 
            header=not os.path.exists('../csv/BETA_INFO.csv'), 
            index=False)

# ==============================================================================
print(f'Creating Downscale data')
Tr = np.array([5, 10, 20, 50, 100, 200])

# shape = (len(df_points['lat_ref']), len(df_points['lon_ref']))
shape = (nlat, nlon)

NYs = np.full([years_num, *shape], np.nan)
CYs = np.full([years_num, *shape], np.nan)
WYs = np.full([years_num, *shape], np.nan)

NYd = np.full([years_num, *shape], np.nan)
CYd = np.full([years_num, *shape], np.nan)
WYd = np.full([years_num, *shape], np.nan)

Mev_d = np.zeros((len(Tr), *shape))
Mev_s = np.zeros((len(Tr), *shape))

BETA = np.zeros([nlat, nlon])
GAMMA = np.zeros([nlat, nlon])

FUNVAL = np.zeros([nlat, nlon])

for la, lo, downres in results:
    
    if len(downres) == 0:

        NYs[:, la, lo] = [np.nan]*years_num
        CYs[:, la, lo] = [np.nan]*years_num
        WYs[:, la, lo] = [np.nan]*years_num
        
        NYd[:, la, lo] = [np.nan]*years_num
        CYd[:, la, lo] = [np.nan]*years_num
        WYd[:, la, lo] = [np.nan]*years_num
        
        Mev_d[:, la, lo] = [np.nan]*len(Tr)
        Mev_s[:, la, lo] = [np.nan]*len(Tr)
        
        BETA[la, lo] = np.nan
        GAMMA[la, lo] = np.nan
        
        FUNVAL[la,lo] = np.nan
    
    else:
    
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
        
        FUNVAL[la,lo] = downres['corr_down_funval']

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
                    "GAMMA": (("lat","lon"), GAMMA),
                    "FUNVAL": (("lat","lon"), FUNVAL)
                    },
    coords={'year':full_years,'Tr':Tr,'lat': lats, 'lon': lons},
    attrs=dict(description=f"Downscaling for '{product}' in the Italy area, using '{param['acf']}' as the acf function, '{param['thresh']} mm' threshold, '{param['corr_method']}' correlation, optimization method '{param['opt_method']}', toll equal '{toll}' and box size '{NEIBHR}x{NEIBHR}'."))

DOWN_xr.NYs.attrs["units"] = "day"
DOWN_xr.NYs.attrs["long_name"] = "Number of wet days"
DOWN_xr.NYs.attrs["origname"] = "Wet days"

DOWN_xr.CYs.attrs["units"] = "dimensionless"
DOWN_xr.CYs.attrs["long_name"] = "Weibull scale parameter"
DOWN_xr.CYs.attrs["origname"] = "Scale"

DOWN_xr.WYs.attrs["units"] = "dimensionless"
DOWN_xr.WYs.attrs["long_name"] = "Weibull shape parameter"
DOWN_xr.WYs.attrs["origname"] = "Shape"

DOWN_xr.Mev_s.attrs["units"] = "mm/day"
DOWN_xr.Mev_s.attrs["long_name"] = "Satellite Maximum Quantiles"
DOWN_xr.Mev_s.attrs["origname"] = "Sat quantiles"

DOWN_xr.NYd.attrs["units"] = "day"
DOWN_xr.NYd.attrs["long_name"] = "Downscale Number of wet days"
DOWN_xr.NYd.attrs["origname"] = "Down wet days"

DOWN_xr.CYd.attrs["units"] = "dimensionless"
DOWN_xr.CYd.attrs["long_name"] = "Downscale Weibull scale parameter"
DOWN_xr.CYd.attrs["origname"] = "Down scale"

DOWN_xr.WYd.attrs["units"] = "dimensionless"
DOWN_xr.WYd.attrs["long_name"] = "Downscale Weibull shape parameter"
DOWN_xr.WYd.attrs["origname"] = "Down shape"

DOWN_xr.Mev_d.attrs["units"] = "mm/day"
DOWN_xr.Mev_d.attrs["long_name"] = "Downscaling Maximum Quantiles"
DOWN_xr.Mev_d.attrs["origname"] = "Downscaling quantiles"

DOWN_xr.BETA.attrs["units"] = "dimensionless"
DOWN_xr.BETA.attrs["long_name"] = "Itermittency function between two generic scales"
DOWN_xr.BETA.attrs["origname"] = "Beta"

DOWN_xr.GAMMA.attrs["units"] = "dimensionless"
DOWN_xr.GAMMA.attrs["long_name"] = "Variance function between two generic scales"
DOWN_xr.GAMMA.attrs["origname"] = "Gamma"

DOWN_xr.FUNVAL.attrs["units"] = "dimensionless"
DOWN_xr.FUNVAL.attrs["long_name"] = "Minimum error achieved by the optimization"
DOWN_xr.FUNVAL.attrs["origname"] = "Funval"

DOWN_xr.lat.attrs["units"] = "degrees_north"
DOWN_xr.lat.attrs["long_name"] = "Latitude"

DOWN_xr.lon.attrs["units"] = "degrees_east"
DOWN_xr.lon.attrs["long_name"] = "Longitude"

# ==============================================================================
DOWN_out = os.path.join('..','output',f'{area}_DOWN_{product}_{time_reso}_{yy_s}_{yy_e}_npix_{param['npix']}_thr_{param['thresh']}_acf_{param['acf']}_{param['opt_method']}_{param['corr_method']}.nc')
print(f'Export Data to {DOWN_out}')
DOWN_xr.to_netcdf(DOWN_out)