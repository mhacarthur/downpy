
import numpy as np
import xarray as xr

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from scipy.interpolate import RBFInterpolator

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

def create_box(DATA_INPUT, clat, clon, npix):
    lats = DATA_INPUT['lat'].data
    lons = DATA_INPUT['lon'].data
    time_vector = DATA_INPUT['time'].values
    
    dset = np.array(DATA_INPUT['PRE'])
    dset[dset<=0]=0

    xrs = xr.DataArray(np.swapaxes(dset, 0, 2),  coords={'lon':lons, 'lat':lats, 'time':time_vector}, dims=('lon', 'lat', 'time'))

    buffer = 0.5*npix*0.1 # To define the limitis of box_3h
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

def wetfrac(array, thresh):
    return np.size(array[array > thresh])/np.size(array)

def space_time_scales_agregations(box_3h, L1, CONDITION, tscales, xscales, npix, thresh):
    
    nlon = len(box_3h['lon'].data)
    nlat = len(box_3h['lat'].data)
    smax = box_3h.shape[0] # max spatial scale

    rainfall_ref = []
    Swet_final = []
    Swet_scale = []
    print(f'Using condition: {CONDITION}')
    print()
    for st in tscales:
        datamat = box_3h.resample(time='{}h'.format(st)).sum(dim='time', skipna = False)

        input_data = datamat.copy()

        print(f'Agregation for time scale: {st} hours')
        for ix, sx in enumerate(xscales):
            if sx == 1:
                print(f'Mean wet fraction for scale: {L1} km')
                rain_tmp = np.zeros([nlon, nlat, input_data.shape[2]])
                wet_tmp = np.zeros([nlon, nlat])
                for i in range(nlon):
                    for j in range(nlat):
                        wet_tmp[i,j] = wetfrac(input_data[i,j,:].data, thresh)
                        rain_tmp[i,j] = input_data[i,j,:].data
                # print(wet_tmp.mean())
                rainfall_ref.append(np.mean(rain_tmp,axis=(0,1)))
                Swet_final.append(wet_tmp.mean())
                Swet_scale.append(L1)

            elif sx == smax:
                print(f'Mean wet fraction for scale: {L1*smax} km')
                rainfall_tmp = input_data.mean(axis=(0,1))
                wet_tmp = wetfrac(rainfall_tmp, thresh)
                # print(wet_tmp)
                rainfall_ref.append(rainfall_tmp.data)
                Swet_final.append(wet_tmp)
                Swet_scale.append(L1*smax)
            
            elif sx > 1 and sx < smax:
                print(f'Mean wet fraction for scale: {L1*sx} km')
                Swet_fraction = []

                if CONDITION == 'OVERLEAP' or CONDITION == 'NONOVERLAP':
                    for i in range(nlon):
                        for j in range(nlat):

                            if CONDITION == 'OVERLEAP':
                                # WITH OVERLAP
                                box_tmp = input_data[i:i+sx,j:j+sx,:]
                            elif CONDITION == 'NONOVERLAP':
                                # WITHOUT OVERLAP
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
        print()

    WET_MATRIX = np.reshape(Swet_final,(10,npix))
    
    return WET_MATRIX

def wet_matrix_extrapolation(WET_MATRIX, spatial_scale, temporal_scale, L1, npix):
    # Create a grid of points for the original data
    original_points = np.array(np.meshgrid(temporal_scale, spatial_scale)).T.reshape(-1, 2)
    wet_fraction_values = WET_MATRIX.ravel()

    # Use RBFInterpolator for cubic extrapolation
    interpolator = RBFInterpolator(original_points, wet_fraction_values, kernel='cubic')

    # New spatial scale with 100 values from 0 to 50 km 
    new_spatial_scale = np.linspace(0, npix * L1, 100)

    # Create new grid for extrapolated data
    new_spatial, new_temporal = np.meshgrid(new_spatial_scale, temporal_scale)

    # Combine the spatial and temporal scales for interpolation
    points_to_interpolate = np.array([new_temporal.ravel(), new_spatial.ravel()]).T

    # Get the interpolated values (including extrapolated values)
    WET_MATRIX_EXTRA = interpolator(points_to_interpolate).reshape(new_temporal.shape)
    
    return WET_MATRIX_EXTRA, new_spatial_scale

def autocorrelation_neighborhood(box_3h, cor_method = 'spearman'):

    xdaily = box_3h.resample(time ='{}h'.format(24)).sum(dim='time', skipna=False).dropna(dim='time', how='any')
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
        for j in range(i+1, nelem):
            tsj = xdaily.dropna(dim='time', how='any').loc[dict(lat=lats9[j], lon=lons9[j])].values
            vdist[count] = haversine(lats9[i], lons9[i], lats9[j], lons9[j])
            if cor_method == 'spearman':
                vcorr[count], _ = spearmanr(tsi, tsj)
            elif cor_method == 'pearson':
                vcorr[count], _ = pearsonr(tsi, tsj)
            count = count + 1

    distance_vector = np.linspace(np.min(vdist), np.max(vdist), 40)

    return vdist, vcorr, distance_vector