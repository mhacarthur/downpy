import os
import numpy as np
import pandas as pd
import xarray as xr

import itertools

from scipy.stats import pearsonr, spearmanr

from scipy.interpolate import RBFInterpolator

from scipy.optimize import differential_evolution
from pathos.multiprocessing import ProcessingPool as Pool

import sys
sys.path.insert(0, os.path.abspath("../function"))
from ART_downscale import fit_yearly_weibull_update, compute_beta, myfun_sse, down_wei

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

def haversine_zorzeto(lat1, lon1, lat2, lon2, convert_to_rad=True):
    def torad(theta):
        return theta*np.pi/180.0
    if convert_to_rad:
        lat1 = torad(lat1)
        lat2 = torad(lat2)
        lon1 = torad(lon1)
        lon2 = torad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371.0 # km
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist = 2*R*np.arctan2( np.sqrt(a), np.sqrt(1-a))
    return dist

def area_lat_lon(lat_c, lon_c, dlat, dlon):
    lat1 = lat_c - dlat/2
    lat2 = lat_c + dlat/2
    lon1 = lon_c - dlon/2
    lon2 = lon_c + dlon/2
    hor_size = haversine(lat1, lon1, lat1, lon2)
    vert_size = haversine(lat1, lon1, lat2, lon1)
    my_area = hor_size*vert_size
    my_edge = np.sqrt(my_area)
    return my_edge, my_area, hor_size, vert_size

def create_box(DATA_INPUT, clat, clon, npix, reso):
    lats = DATA_INPUT['lat'].data
    lons = DATA_INPUT['lon'].data
    time_vector = DATA_INPUT['time'].values
    
    dset = np.array(DATA_INPUT['PRE'])
    dset[dset<=0]=0

    xrs = xr.DataArray(np.swapaxes(dset, 0, 2),  coords={'lon':lons, 'lat':lats, 'time':time_vector}, dims=('lon', 'lat', 'time'))

    buffer = 0.5*npix*reso # To define the limitis of box_3h
    eps = 1e-4 # to make sure to include limtis -> add an eps buffer
    solat = clat - buffer + eps
    nolat = clat + buffer + eps
    ealon = clon + buffer + eps
    welon = clon - buffer + eps
    bcond = np.logical_and(
                np.logical_and( xrs.lat > solat, xrs.lat < nolat),
                np.logical_and( xrs.lon > welon, xrs.lon < ealon))

    # Box 3h time resolution
    box_3h = xrs.where(bcond, drop = True).load()

    nlon = len(box_3h['lon'].data)
    nlat = len(box_3h['lat'].data)

    if nlon != nlat:
        print('downscale warning: box sizes are not equal')
    if nlon % 2 == 0:
        print('downscale warning: at least one box size has even length')

    return box_3h, bcond

def create_box_v2(DATA_INPUT, clat, clon, npix):
    '''
    Create a square box around a central point (clat, clon) with npix pixels.
    The npix is the number of pixel to the left and right of the central point.
    npix = 1, the box is 3x3 pixels.
    npix = 2, the box is 5x5 pixels.
    npix = 3, the box is 7x7 pixels.
    DATA_INPUT: xarray dataset with lat and lon coordinates [xarray.core.dataset.Dataset].
    clat: central latitude [np.float].
    clon: central longitude [np.float].
    npix: number of pixels to the left and right of the central point [np.int].
    '''
    lats = DATA_INPUT['lat'].data
    lons = DATA_INPUT['lon'].data

    lat_idx = np.abs(lats - clat).argmin()
    lon_idx = np.abs(lons - clon).argmin()

    lat_start = max(0, lat_idx - npix)
    lat_end = min(len(lats) - 1, lat_idx + npix)
    lon_start = max(0, lon_idx - npix)
    lon_end = min(len(lons) - 1, lon_idx + npix)

    box = DATA_INPUT.isel(
    lat=slice(lat_start, lat_end + 1),  # +1 because slice is exclusive at the end
    lon=slice(lon_start, lon_end + 1)
    )

    return box

def wetfrac(array, thresh):
    if len(array) == 0:
        return np.nan
    else:
        return np.size(array[array > thresh])/np.size(array)

def space_time_scales_agregations(box_3h, L1, CONDITION, tscales, xscales, npix, thresh):
    nlon = len(box_3h['lon'].data)
    nlat = len(box_3h['lat'].data)
    smax = box_3h.shape[0] # max spatial scale

    rainfall_ref = []
    Swet_final = []
    Swet_scale = []
    # print(f'Scales agregation condition: {CONDITION}')
    # print()
    for st in tscales:
        datamat = box_3h.resample(time='{}h'.format(st)).sum(dim='time', skipna = False)

        input_data = datamat.copy()

        # print(f'Agregation for time scale: {st} hours')
        for ix, sx in enumerate(xscales):
            if sx == 1:
                # print(f'Mean wet fraction for scale: {L1} km')
                rain_tmp = np.zeros([nlon, nlat, input_data.shape[2]])
                wet_tmp = np.zeros([nlon, nlat])
                for i in range(nlon):
                    for j in range(nlat):
                        wet_tmp[i,j] = wetfrac(input_data[i,j,:].data, thresh)
                        rain_tmp[i,j] = input_data[i,j,:].data
                rainfall_ref.append(np.mean(rain_tmp,axis=(0,1)))
                Swet_final.append(wet_tmp.mean())
                Swet_scale.append(L1)

            elif sx == smax:
                # print(f'Mean wet fraction for scale: {L1*smax} km')
                rainfall_tmp = input_data.mean(axis=(0,1))
                wet_tmp = wetfrac(rainfall_tmp, thresh)
                rainfall_ref.append(rainfall_tmp.data)
                Swet_final.append(wet_tmp)
                Swet_scale.append(L1*smax)
            
            elif sx > 1 and sx < smax:
                # print(f'Mean wet fraction for scale: {L1*sx} km')
                Swet_fraction = []

                if CONDITION == 'OVERLEAP' or CONDITION == 'NONOVERLAP':
                    for i in range(nlon):
                        for j in range(nlat):

                            if CONDITION == 'OVERLEAP': # WITH OVERLAP
                                box_tmp = input_data[i:i+sx,j:j+sx,:]
                            elif CONDITION == 'NONOVERLAP': # WITHOUT OVERLAP
                                box_tmp = input_data[sx*i:sx*i+sx,sx*j:sx*j+sx,:]

                            if box_tmp.shape[0] == sx and box_tmp.shape[1] == sx:
                                wet_tmp = wetfrac(box_tmp.mean(axis=(0,1)).data, thresh)
                                Swet_fraction.append(wet_tmp)

                elif CONDITION == 'FOCUS': # FOR AGREGATION WITH FOCUS IN SPECIFIC POINT (NOW ONLY WORK FOR npix = 5)
                    L = sx - 1
                    if sx == 2 or sx == 3:
                        origin_x_pos, origin_y_pos = 2, 2
                        for loop in range(4):
                            if loop == 0:
                                box_tmp = input_data[origin_x_pos-1*L:origin_x_pos+1,origin_y_pos-1*L:origin_y_pos+1]
                            elif loop == 1:
                                box_tmp = input_data[origin_x_pos:origin_x_pos+2*L,origin_y_pos-1*L:origin_y_pos+1]
                            elif loop == 2:
                                box_tmp = input_data[origin_x_pos-1*L:origin_x_pos+1,origin_y_pos:origin_y_pos+2*L]
                            elif loop == 3:
                                box_tmp = input_data[origin_x_pos:origin_x_pos+2*L,origin_y_pos:origin_y_pos+2*L]
                            wet_tmp = wetfrac(box_tmp.mean(axis=(0,1)).data, thresh)
                            Swet_fraction.append(wet_tmp)

                    elif sx == 4:
                        for loop in range(4):
                            if loop == 0:
                                origin_x_pos, origin_y_pos = 3, 3
                                box_tmp = input_data[origin_x_pos-1*L:origin_x_pos+1,origin_y_pos-1*L:origin_y_pos+1]
                            elif loop == 1:
                                origin_x_pos, origin_y_pos = 1, 3
                                tmp_pos_RB = input_data[origin_x_pos:origin_x_pos+2*L,origin_y_pos-1*L:origin_y_pos+1]  
                            elif loop == 2:
                                origin_x_pos, origin_y_pos = 3, 1
                                tmp_pos_LT = input_data[origin_x_pos-1*L:origin_x_pos+1,origin_y_pos:origin_y_pos+2*L]
                            elif loop == 3:
                                origin_x_pos, origin_y_pos = 1, 1
                                tmp_pos_RT = input_data[origin_x_pos:origin_x_pos+2*L,origin_y_pos:origin_y_pos+2*L]
                            wet_tmp = wetfrac(box_tmp.mean(axis=(0,1)).data, thresh)
                            Swet_fraction.append(wet_tmp)

                rainfall_ref.append(box_tmp.sum(axis=(0,1)).data) # Rainfall series
                Swet_final.append(np.mean(Swet_fraction)) # wet fraction
                Swet_scale.append(L1*sx) # Spatial scales
        # print()
    # print()

    WET_MATRIX = np.reshape(Swet_final,(len(tscales),npix))

    return WET_MATRIX

def space_time_scales_agregations_v2(box, time_reso, tscales, xscales, npix, thresh):
    nlon = len(box['lon'].data)
    nlat = len(box['lat'].data)
    smax = box.shape[0]
    Swet_final = []

    for st in tscales:
        # if time_reso == '3h':
        input_data = box.resample(time='{}h'.format(st)).sum(dim='time', skipna = False)
        # elif time_reso == '1dy':
        #     input_data = box.resample(time='{}d'.format(st)).sum(dim='time', skipna = False)

        for ix, sx in enumerate(xscales):
            if sx == 1:
                wet_tmp = np.zeros([nlon, nlat])
                for i in range(nlon):
                    for j in range(nlat):
                        wet_tmp[i,j] = wetfrac(input_data[i,j,:].data, thresh)
                Swet_final.append(np.nanmean(wet_tmp))

            elif sx == smax:
                rainfall_tmp = input_data.mean(axis=(0,1))
                wet_tmp = wetfrac(rainfall_tmp, thresh)
                Swet_final.append(np.nanmean(wet_tmp))

            elif sx > 1 and sx < smax:
                Swet_fraction = []
                # ================================================================================
                for i in range(nlon):
                    for j in range(nlat):
                        box_tmp = input_data[i:i+sx,j:j+sx,:]
                        if box_tmp.shape[0] == sx and box_tmp.shape[1] == sx:
                            wet_tmp = wetfrac(np.nanmean(box_tmp.data,axis=(0,1)), thresh)
                            Swet_fraction.append(wet_tmp)
                # ================================================================================
                # c1 = np.zeros(4)
                # c1[0] = wetfrac(input_data[:sx, :sx, :].mean(dim=('lat', 'lon'),
                #                 skipna=False).dropna(dim='time', how='any'),
                #                 thresh)
                # c1[1] = wetfrac(input_data[-sx:, :sx, :].mean(dim=('lat', 'lon'),
                #                 skipna=False).dropna(dim='time', how='any'),
                #                 thresh)
                # c1[2] = wetfrac(input_data[:sx, :sx, :].mean(dim=('lat', 'lon'),
                #                 skipna=False).dropna(dim='time', how='any'),
                #                 thresh)
                # c1[3] = wetfrac(input_data[-sx:, :sx, :].mean(dim=('lat', 'lon'),
                #                 skipna=False).dropna(dim='time', how='any'),
                #                 thresh)
                # Swet_fraction.append(np.mean(c1))
                # ================================================================================

                Swet_final.append(np.nanmean(Swet_fraction))

    WET_MATRIX = np.reshape(Swet_final,(len(tscales),npix))
    
    return WET_MATRIX

def wet_matrix_extrapolation(WET_MATRIX, spatial_scale, temporal_scale, L1, npix):
    # Create a grid of points for the original data
    original_points = np.array(np.meshgrid(temporal_scale, spatial_scale)).T.reshape(-1, 2)
    wet_fraction_values = WET_MATRIX.ravel()

    # Use RBFInterpolator for cubic extrapolation
    interpolator = RBFInterpolator(original_points, wet_fraction_values, kernel='cubic')

    # New spatial scale with 100 values from 0 to 50 km 
    new_spatial_scale = np.linspace(0, (2*npix+1)*L1, 100)

    # Create new grid for extrapolated data
    new_spatial, new_temporal = np.meshgrid(new_spatial_scale, temporal_scale)

    # Combine the spatial and temporal scales for interpolation
    points_to_interpolate = np.array([new_temporal.ravel(), new_spatial.ravel()]).T

    # Get the interpolated values (including extrapolated values)
    WET_MATRIX_EXTRA = interpolator(points_to_interpolate).reshape(new_temporal.shape)
    
    return WET_MATRIX_EXTRA, new_spatial_scale

def autocorrelation_neighborhood(box, time_reso, t_target, thresh, cor_method = 'pearson'):
    if time_reso == '3h':
        xdaily = box.resample(time ='{}h'.format(t_target)).sum(dim='time', skipna=False).dropna(dim='time', how='any')
    elif time_reso == '1dy':
        xdaily = box
    else:
        print(f'Erorr: {time_reso} not valid')
        return None

    lats = xdaily.dropna(dim='time', how='any').lat.values
    lons = xdaily.dropna(dim='time', how='any').lon.values
    nlats = np.size(lats)
    nlons = np.size(lons)
    nelem = nlats*nlons
    lats9 = np.repeat(lats, nlons)
    lons9 = np.tile(lons, nlats)

    ncorr = (nelem)*(nelem - 1)//2
    vdist = np.zeros(ncorr)
    count = 0

    vcorr = np.zeros(ncorr)

    for i in range(nelem):
        tsi = xdaily.dropna(dim='time', how='any').loc[dict(lat=lats9[i], lon=lons9[i])].values
        tsi = np.maximum(tsi-thresh, 0.0)
        for j in range(i+1, nelem):
            tsj = xdaily.dropna(dim='time', how='any').loc[dict(lat=lats9[j], lon=lons9[j])].values
            tsj = np.maximum(tsj-thresh, 0.0)
            vdist[count] = haversine(lats9[i], lons9[i], lats9[j], lons9[j])
            if cor_method == 'spearman':
                vcorr[count], _ = spearmanr(tsi, tsj)
            elif cor_method == 'pearson':
                vcorr[count], _ = pearsonr(tsi, tsj)
            count = count + 1

    distance_vector = np.linspace(np.min(vdist), np.max(vdist), 40)

    return vdist, vcorr, distance_vector

def autocorrelation_neighborhood_v2(box, time_reso, t_target, thresh, cor_method = 'pearson'):
    if time_reso == '3h':
        xdaily = box.resample(time ='{}h'.format(t_target)).sum(dim='time', skipna=False).dropna(dim='time', how='any')
    elif time_reso == '1dy':
        xdaily = box
    else:
        print(f'Erorr: {time_reso} not valid')
        return None

    lats = xdaily.dropna(dim='time', how='any').lat.values
    lons = xdaily.dropna(dim='time', how='any').lon.values
    nlat = np.size(lats)
    nlon = np.size(lons)

    points = list(itertools.combinations([(i, j) for i in range(nlat) for j in range(nlon)], 2))

    vdist = []
    vcorr = []

    for (lat1, lon1), (lat2, lon2) in points:
        # Extraer las series temporales de cada punto
        p1 = xdaily['PRE'][:, lat1, lon1].values
        p1 = np.maximum(p1-thresh, 0.0) # original
        # p1 = p1[p1 > thresh] - thresh
        
        p2 = xdaily['PRE'][:, lat2, lon2].values
        p2 = np.maximum(p2-thresh, 0.0) # original
        # p2 = p2[p2 > thresh] - thresh

        # Eliminar NaNs en ambos arrays
        mask = ~np.isnan(p1) & ~np.isnan(p2)
        p1_clean, p2_clean = p1[mask], p2[mask]

        if len(p1_clean) <= 3 and len(p2_clean) <= 3:
            corr = np.nan
            dist = np.nan
        else:
            dist = haversine(
                            box['lat'].values[lat1], 
                            box['lat'].values[lon1], 
                            box['lat'].values[lat2], 
                            box['lat'].values[lon2])

            if cor_method == 'spearman':
                corr = spearmanr(p1_clean, p2_clean)[0]
            elif cor_method == 'pearson':
                corr = pearsonr(p1_clean, p2_clean)[0]
            else:
                print(f'ERROR method {cor_method} not found')
                return None

        vcorr.append(float(corr))
        vdist.append(float(dist))

    return vdist, vcorr

def spatial_correlation(DF_input, threshold, dir_base, cor_method):
    count = 0
    correlation = []
    distance = []
    names1 = []
    names2 = []

    for ii in range(len(DF_input)):
        serie1 = pd.read_csv(os.path.join(dir_base, 'CLEAR_1dy', f'{DF_input['File_Name'].values[ii]}.csv'))
        serie1['TIME'] = pd.to_datetime(serie1['TIME'])
        pos1 = ([DF_input['Lat'].values[ii], DF_input['Lon'].values[ii]])
        name1 = DF_input['File_Name'].values[ii]

        for jj in range(count, len(DF_input)):
            if jj == ii:
                pass
            else:
                serie2 = pd.read_csv(os.path.join(dir_base, 'CLEAR_1dy', f'{DF_input['File_Name'].values[jj]}.csv'))
                serie2['TIME'] = pd.to_datetime(serie2['TIME'])
                pos2 = ([DF_input['Lat'].values[jj], DF_input['Lon'].values[jj]])
                name2 = DF_input['File_Name'].values[jj]

                names1.append(name1)
                names2.append(name2)

                df_common = pd.merge(serie1, serie2, on='TIME', suffixes=('_series1', '_series2'))
                df_common_no_nan = df_common.dropna(subset=['PRE_series1', 'PRE_series2'])
                
                tsi = np.maximum(df_common_no_nan['PRE_series1'].values-threshold, 0.0)
                tsj = np.maximum(df_common_no_nan['PRE_series2'].values-threshold, 0.0)
                # print(len(tsi), len(tsj))

                if len(tsi) <= 1 or len(tsi) <= 1:
                    corr = np.nan
                else:
                    if cor_method == 'spearman':
                        corr, _ = spearmanr(tsi, tsj)
                    elif cor_method == 'pearson':
                        corr, _ = pearsonr(tsi, tsj)
                    else:
                        # print(f'ERROR method {cor_method} not found')
                        raise ValueError(f"Unsupported correlation type '{cor_method}'. Please use 'pearson' or 'spearman'.")


                correlation.append(corr)

                ## WARNING!
                ## Estas distancias no estan verificadas, si usan lat y lon o solo indices
                # dist = haversine(pos1[0], pos1[1], pos2[0], pos2[1])
                dist = haversine_zorzeto(pos1[0], pos1[1], pos2[0], pos2[1])
                distance.append(dist)

        count = count + 1

    CORR_DF = pd.DataFrame({'name1': names1, 'name2':names2, 'dist':distance, 'corr':correlation})
    CORR_DF = CORR_DF.sort_values(by='dist').reset_index(drop=True)
    
    return CORR_DF

# def ART_downscalling(DATA_in, time_reso, lats, lons, lat_c, lon_c, PARAM):
#     PRE_daily = DATA_in.resample(time ='D').sum(dim='time', skipna=False)

#     DATES_daily = PRE_daily['time']

#     i_ = np.where(lats==lat_c)[0][0]
#     j_ = np.where(lons==lon_c)[0][0]

#     IMERG_pixel_1dy = PRE_daily['PRE'][:,i_,j_].data

#     IMERG_pixel_1dy_xr = xr.DataArray(
#                 IMERG_pixel_1dy, 
#                 coords={'time':PRE_daily['time'].values}, 
#                 dims=('time'))

#     IMERG_WEIBULL_YEAR = fit_yearly_weibull_update(
#                 IMERG_pixel_1dy_xr, 
#                 thresh=PARAM['thresh'], 
#                 maxmiss=PARAM['maxmiss'])

#     box_3h, bcond = create_box(DATA_in, lat_c, lon_c, PARAM['npix'], reso=PARAM['radio'])

#     smax = box_3h.shape[0] # max spatial scale
#     tscales = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36, 48, 96])*PARAM['dt']
#     tscales = tscales[tscales < PARAM['tmax'] + 0.001]
#     xscales = np.arange(1, smax+1)
#     xscales_km = xscales*PARAM['L1']
#     ntscales = np.size(tscales)
#     nsscales = np.size(xscales)

#     WET_MATRIX = space_time_scales_agregations_v2(box_3h, time_reso, PARAM['L1'], tscales, xscales, 2*PARAM['npix']+1, PARAM['thresh'])
#     # WET_MATRIX = space_time_scales_agregations(
#     #             box_3h, 
#     #             PARAM['L1'], 
#     #             PARAM['condition'], 
#     #             tscales, 
#     #             xscales, 
#     #             PARAM['npix'], 
#     #             PARAM['thresh'])

#     xscales_km_2d, tscales_2d = np.meshgrid(xscales_km, tscales)
#     ntscales = np.size(tscales)
#     nxscales = np.size(xscales)

#     tscales_INTER = np.linspace(np.min(tscales), np.max(tscales), PARAM['ninterp'])
#     WET_MATRIX_INTER = np.zeros((PARAM['ninterp'], nxscales))
    
#     for col in range(nxscales):
#         WET_MATRIX_INTER[:, col] = np.interp(tscales_INTER, tscales, WET_MATRIX[:, col])

#     WET_MATRIX_EXTRA, new_spatial_scale = wet_matrix_extrapolation(
#                 WET_MATRIX_INTER, 
#                 xscales_km, 
#                 tscales_INTER, 
#                 PARAM['L1'], 
#                 PARAM['npix'])

#     origin_ref = [PARAM['origin_x'], PARAM['origin_t']]
#     target_ref = [PARAM['target_x'], PARAM['target_t']]

#     beta = compute_beta(WET_MATRIX_EXTRA, origin_ref, target_ref, new_spatial_scale, tscales_INTER)

#     vdist, vcorr, distance_vector = autocorrelation_neighborhood(
#                 box_3h, 
#                 t_target = PARAM['target_t'], 
#                 thresh = PARAM['thresh'], 
#                 cor_method = PARAM['corr_method'])

#     # FIT, _ = curve_fit(str_exp_fun, vdist, vcorr)
#     # FIT_d0, FIT_mu0 = FIT
#     # FIT, _ = curve_fit(epl_fun, vdist, vcorr)
#     # FIT_eps, FIT_alp = FIT

#     vdist_sorted = np.sort(vdist) # order distance
#     vcorr_sorted = vcorr[np.argsort(vdist)] # order correlation in relation to distance
#     toll_cluster = 0.5

#     cluster = np.zeros(len(vdist_sorted))
#     count = 0
#     for i in range(1, len(vdist_sorted)):
#         if np.abs(vdist_sorted[i]-vdist_sorted[i-1]) < toll_cluster:
#             cluster[i] = count
#         else:
#             count = count + 1
#             cluster[i] = count

#     clust = set(cluster) # Extract only the uniques values
#     nclust = len(clust) # Numero de grupos

#     vdist_ave = np.zeros(nclust)
#     vcorr_ave = np.zeros(nclust)
#     for ei, elem in enumerate(clust):
#         di = vdist_sorted[cluster==elem] # Distance
#         ci = vcorr_sorted[cluster==elem] # Correlation
#         vdist_ave[ei] = np.mean(di) # Mean Distance
#         vcorr_ave[ei] = np.mean(ci) # Mean Correlation

#     # FIT, _ = curve_fit(epl_fun, vdist_ave, vcorr_ave)
#     # FIT_ave_eps, FIT_ave_alp = FIT

#     # bounds = [(0.0, 200),(0, 1)] # ORIGINAL LIMITS BY ZORZETO
#     bounds = [(0.0, 25.0),(0, 0.3)] # NEW LIMITS USING ALL CORRELATIONS FUNCTION IN VENETO
    
    def myfun(pardown):
        return myfun_sse(vdist_ave, vcorr_ave, pardown, PARAM['L1'], acf=PARAM['acf'])

    with Pool(nodes=PARAM['cores']) as pool:
        resmin = differential_evolution(
            myfun,
            bounds,
            disp=True,
            tol=0.05,
            atol=0.05,
            workers=pool.map
        )

    param1 = resmin.x[0]
    param2 = resmin.x[1]

    NYd, CYd, WYd, gamYd, _ = down_wei(
                        IMERG_WEIBULL_YEAR[:,0], 
                        IMERG_WEIBULL_YEAR[:,1], 
                        IMERG_WEIBULL_YEAR[:,2], 
                        PARAM['L1'], 
                        PARAM['L0'], 
                        beta, 
                        (param1, param2), 
                        acf=PARAM['acf'])

    DOWN_WEIBULL_YY = np.zeros([len(NYd), 3])
    DOWN_WEIBULL_YY[:,0] = NYd
    DOWN_WEIBULL_YY[:,1] = CYd
    DOWN_WEIBULL_YY[:,2] = WYd

    dict_out = dict({'beta':beta, 'gamma':gamYd, 'param1':param1, 'param2':param2})

    return IMERG_WEIBULL_YEAR, DOWN_WEIBULL_YY, dict_out

def relative_error(data1, data2):
    ii, jj = data1.shape[0], data1.shape[1]
    ERROR = np .zeros([ii, jj])*np.nan
    for i in range(ii):
        for j in range(jj):
            diff = (data1[i,j] - data2[i,j])/data1[i,j]
            ERROR[i,j] = diff
    return ERROR