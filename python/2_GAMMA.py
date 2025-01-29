import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import xarray as xr

import psutil
from joblib import Parallel, delayed

from scipy.optimize import differential_evolution

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath(".."))
from function import ART_downscale as ART_down
from function import ART_preprocessing as ART_pre

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

NEIBHR = 2*param['npix']+1

print(f'Read json file   : {json_read.split('/')[-1]}')
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

if product == 'MSWEP' or product == 'PERSIANN' or product == 'SM2RAIN':
    PRE_data = PRE_data.sel(lat=slice(lat_max+1.5, lat_min-1.5), lon=slice(lon_min-1.5, lon_max+1.5))
else:
    PRE_data = PRE_data.sel(lat=slice(lat_min-1.5, lat_max+1.5), lon=slice(lon_min-1.5, lon_max+1.5))

lats = PRE_data['lat'].data
lons = PRE_data['lon'].data

lon2d, lat2d = np.meshgrid(lons, lats)

nlon = np.size(lons)
nlat = np.size(lats)
ntime = len(PRE_data['time'])

# =============================================================================
if product == 'MSWEP' or product == 'PERSIANN' or product == 'SM2RAIN':
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
def gamma_3h_1dy(DATA_in, time_reso, lat_c, lon_c, PARAM):
    if time_reso == '3h':
        PRE_daily = DATA_in.resample(time ='D').sum(dim='time', skipna=False)
        BOX = ART_pre.create_box_v2(PRE_daily, lat_c, lon_c, PARAM['npix'])
    elif time_reso == '1dy':
        BOX = ART_pre.create_box_v2(DATA_in, lat_c, lon_c, PARAM['npix'])
        PRE_daily = DATA_in
    else:
        print(f'Erorr: {time_reso} not valid')
        return None

    print(f"Neighborhood area: {BOX.sizes['lat']}x{BOX.sizes['lon']}")
    
    i_ = np.where(lats==lat_c)[0][0]
    j_ = np.where(lons==lon_c)[0][0]

    IMERG_pixel_1dy = PRE_daily['PRE'][:,i_,j_].data
    IMERG_pixel_1dy_xr = xr.DataArray(
                IMERG_pixel_1dy, 
                coords={'time':PRE_daily['time'].values}, 
                dims=('time'))

    IMERG_WEIBULL_YEAR = ART_down.fit_yearly_weibull_update(
                    IMERG_pixel_1dy_xr, 
                    thresh=PARAM['thresh'], 
                    maxmiss=PARAM['maxmiss'])
    
    vdist, vcorr = ART_pre.autocorrelation_neighborhood_v2(
                    BOX, 
                    time_reso, 
                    param['target_t'], 
                    param['thresh'])
    
    mask = ~np.isnan(vdist) & ~np.isnan(vcorr)
    vdist, vcorr = np.array(vdist)[mask], np.array(vcorr)[mask]

    vdist_sorted = np.sort(vdist) # order distance
    vcorr_sorted = vcorr[np.argsort(vdist)] # order correlation in relation to distance
    toll_cluster = 0.5

    cluster = np.zeros(len(vdist_sorted))
    count = 0
    for i in range(1, len(vdist_sorted)):
        if np.abs(vdist_sorted[i]-vdist_sorted[i-1]) < toll_cluster:
            cluster[i] = count
        else:
            count = count + 1
            cluster[i] = count

    clust = set(cluster) # Extract only the uniques values
    nclust = len(clust) # Numero de grupos

    vdist_ave = np.zeros(nclust)
    vcorr_ave = np.zeros(nclust)
    for ei, elem in enumerate(clust):
        di = vdist_sorted[cluster==elem] # Distance
        ci = vcorr_sorted[cluster==elem] # Correlation
        vdist_ave[ei] = np.mean(di) # Mean Distance
        vcorr_ave[ei] = np.mean(ci) # Mean Correlation
    
    bounds = [(0.0, 25.0),(0, 0.3)] # NEW LIMITS USING ALL CORRELATIONS FUNCTION IN VENETO

    def myfun(pardown):
        return ART_down.myfun_sse(vdist_ave, vcorr_ave, pardown, PARAM['L1'], acf=PARAM['acf'])

    resmin = differential_evolution(
                myfun,
                bounds,
                disp=True,
                tol=0.03,
                atol=0.03,
                workers=1,
                updating='deferred'
            )

    param1 = resmin.x[0]
    param2 = resmin.x[1]

    gamma = ART_down.gamma_manual(IMERG_WEIBULL_YEAR[:,0], 
                        IMERG_WEIBULL_YEAR[:,1], 
                        IMERG_WEIBULL_YEAR[:,2], 
                        PARAM['L1'], 
                        PARAM['L0'], 
                        (param1, param2), 
                        acf=PARAM['acf'])

    return gamma

# =============================================================================
def compute_for_point(lat_idx, lon_idx):
    return gamma_3h_1dy(PRE_data, time_reso, lats[lat_idx], lons[lon_idx], param)

Resource = []

start_time = time.time()

results = Parallel(n_jobs=param['GAMMA_cores'])(
    delayed(compute_for_point)(la, lo) for la in ndices_lat for lo in ndices_lon
    )

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
memory_consumed = psutil.virtual_memory().used / 1024 ** 3

print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
print(f"Memory consumed: {memory_consumed:.3f} GB")

# =============================================================================
INFO = pd.DataFrame({
                    'Product':product,
                    'Parameter':['GAMMA'],
                    'Resolution_t':[time_reso],
                    'Neighborhood':[NEIBHR],
                    'Cores':param['BETA_cores'],
                    'Time(min)':np.round(elapsed_minutes,3)})

INFO.to_csv('../csv/GAMMA_INFO.csv', 
            mode='a', 
            header=not pd.io.common.file_exists('../csv/GAMMA_INFO.csv'), 
            index=False)

# =============================================================================
GAMMA_VENETO = np.array(results).reshape(len(ndices_lat), len(ndices_lon))

GAMMA_xr = xr.Dataset(data_vars={"GAMMA": (("lat","lon"), GAMMA_VENETO.data)},
                    coords={'lat': lats[ndices_lat], 'lon': lons[ndices_lon]},
                    attrs=dict(description=f"GAMMA of {product} for Veneto region limited as 10.5E to 13.5E and 44.5N to 47N"))

GAMMA_xr.GAMMA.attrs["units"] = "dimensionless"
GAMMA_xr.GAMMA.attrs["long_name"] = "Relation between Origin and Tarjet wet fraction"
GAMMA_xr.GAMMA.attrs["origname"] = "GAMMA"

GAMMA_xr.lat.attrs["units"] = "degrees_north"
GAMMA_xr.lat.attrs["long_name"] = "Latitude"

GAMMA_xr.lon.attrs["units"] = "degrees_east"
GAMMA_xr.lon.attrs["long_name"] = "Longitude"

GAMMA_out = os.path.join('..','output',f'VENETO_GAMMA_{product}_{time_reso}_{yy_s}_{yy_e}.nc')
print(f'Export PRE data to {GAMMA_out}')
GAMMA_xr.to_netcdf(GAMMA_out)

print(f'GAMMA data saved in {GAMMA_out}')