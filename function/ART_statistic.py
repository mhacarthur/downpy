import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cluster import KMeans

# Root-Mean-Square Deviation (RMSD)
def calculate_rmsd(obs, mod):
    differences = mod - obs
    squared_differences = np.square(differences)
    mean_squared_differences = np.nanmean(squared_differences, axis=0)
    rmsd = np.sqrt(mean_squared_differences)
    return rmsd

# Mean Bias Error (MBE)
def calculate_mbe(obs, mod):
    return np.nanmean(mod - obs,axis=0)

# Mean Absolute Error (MAE)
def calculate_mae(obs, mod):
    return np.nanmean(np.abs(mod - obs),axis=0)

# Mean Squared Error (MSE)
def calculate_mse(obs, mod):
    return np.nanmean((mod - obs) ** 2,axis=0)

def CDFt(obs, sat):
    """
    Aplica la corrección de sesgo usando CDFt.
    Si se proporciona 'sat_future', también ajusta datos futuros.
    """
    # CDF empírica de los datos observados y satelitales
    obs_sorted = np.sort(obs)
    
    # Transformación de cuantiles
    sat_percentiles = stats.rankdata(sat, method='average') / len(sat)
    corrected_sat = np.interp(sat_percentiles, np.linspace(0,1,len(obs_sorted)), obs_sorted)
    
    return corrected_sat

def ISIMIP_QM(obs, sat):
    """
    Aplica la corrección de sesgo usando ISIMIP QM (Quantile Mapping).
    """
    quantiles = np.linspace(0, 1, len(obs))
    obs_quantiles = np.quantile(obs, quantiles)
    sat_quantiles = np.quantile(sat, quantiles)
    corrected_sat = np.interp(sat, sat_quantiles, obs_quantiles)

    return corrected_sat

def ISIMIP_QM_ALL(obs, sat):
    quantiles = np.linspace(0, 1, len(obs))
    obs_quantiles = np.quantile(obs, quantiles)
    sat_quantiles = np.quantile(sat, quantiles)
    corrected_sat = np.interp(sat, sat_quantiles, obs_quantiles)

    return corrected_sat

def extract_all_quantiles(product):
    dir_base = os.path.join('/','media','arturo','T9','Data','Italy')
    hdf5_file = os.path.join(dir_base,'statistics',f'statistics_obs_{product}.h5')
    data = pd.HDFStore(hdf5_file, mode='r')

    all_keys = data.keys()
    all_QUANTILES = [k for k in all_keys if k.endswith("/QUANTILES")]
    
    RE_raw = []
    RE_down = []

    for nn in range(len(all_QUANTILES)):
        DICT = data[all_QUANTILES[nn]]
        RE_raw_ = DICT.RE_raw.values[3]
        RE_down_ = DICT.RE_down.values[3]
    
        RE_raw.append(RE_raw_)
        RE_down.append(RE_down_)
    
    RE_raw = np.array(RE_raw)
    RE_down = np.array(RE_down)
    
    RE_raw = np.where(RE_raw >= 1.3, np.nan, RE_raw)
    RE_down = np.where(RE_down >= 1.3, np.nan, RE_down)
    
    return RE_raw, RE_down

def get_relative_error(product, dir_base, val_max=1.1):

    list_remove = ['IT-820_1424_FTS_1440_QCv4.csv', 'IT-250_602781_FTS_1440_QCv4.csv', 'IT-250_602779_FTS_1440_QCv4.csv', 'IT-780_2370_FTS_1440_QCv4.csv', 'IT-750_450_FTS_1440_QCv4.csv']

    hdf5_file = os.path.join(dir_base,'statistics',f'statistics_obs_{product}.h5')
    data = pd.HDFStore(hdf5_file, mode='r')

    keys = data.keys()
    keys_QUANTILES = [k for k in keys if k.endswith("/QUANTILES")]
    keys_INFO = [k for k in keys if k.endswith('/INFO')]

    stations = []
    lats, lons, elevs = [], [], []
    RED, REDn = [], []
    RER, RERn = [], []
    for nn in range(len(keys_INFO)):
        station = keys_INFO[nn].split('/')[2]
        
        if station in list_remove:
            continue
        else:
            lat = data[keys_INFO[nn]]['lat_obs'].values[0]
            lon = data[keys_INFO[nn]]['lon_obs'].values[0]
            elev = data[keys_INFO[nn]]['elev_obs'].values[0]
            RED_ = data[keys_QUANTILES[nn]].RE_down.values[3]
            RER_ = data[keys_QUANTILES[nn]].RE_raw.values[3]

            stations.append(station)
            lats.append(lat)
            lons.append(lon)
            elevs.append(elev)
            RED.append(RED_)
            RER.append(RER_)

    REDn = (RED - np.nanmin(RED))/(np.nanmax(RED) - np.nanmin(RED))
    RERn = (RER - np.nanmin(RER))/(np.nanmax(RER) - np.nanmin(RER))

    DF_DATA = pd.DataFrame({'STATION':stations, 'LON':lons, 'LAT':lats, 'ELEV':elevs, 'RER':RER, 'RERn':RERn, 'RED':RED, 'REDn':REDn})
    DF_DATA.loc[DF_DATA['RER'] > val_max, 'RER'] = np.nan
    DF_DATA.loc[DF_DATA['RER'].isna(), 'RED'] = np.nan

    return DF_DATA

def DF_elevation(DF):
    group_colors = {'≤25%':  '#2c7bb6','25–50%': '#abd9e9','50–75%': '#fdae61','>75%':   '#d7191c'}
    
    Elevation_norm = (DF.ELEV.values - np.min(DF.ELEV.values)) / (np.max(DF.ELEV.values) - np.min(DF.ELEV.values))

    DF['ELEVn'] = Elevation_norm
    DF['ELEV_QUARTILE'] = pd.qcut(DF['ELEV'],4,labels=['≤25%', '25–50%', '50–75%', '>75%'],duplicates='drop')
    DF['ELEV_QUARTILEn'] = pd.qcut(DF['ELEV'],q=4,labels=[1, 2, 3, 4]).astype(int)
    DF['ELEV_color'] = DF['ELEV_QUARTILE'].map(group_colors)
    
    GROUP1 = DF[DF['ELEV_QUARTILEn']==1]
    GROUP2 = DF[DF['ELEV_QUARTILEn']==2]
    GROUP3 = DF[DF['ELEV_QUARTILEn']==3]
    GROUP4 = DF[DF['ELEV_QUARTILEn']==4]
    
    mean_g1 = np.nanmean(GROUP1.RED)
    mean_g2 = np.nanmean(GROUP2.RED)
    mean_g3 = np.nanmean(GROUP3.RED)
    mean_g4 = np.nanmean(GROUP4.RED)
    
    DF_ALL = [GROUP1, GROUP2, GROUP3, GROUP4]
    means = [mean_g1, mean_g2, mean_g3, mean_g4]
    
    return DF_ALL, means

def elevation_kmeans_robusto(DF_input):
    """
    Versión robusta con verificación completa
    """
    DF = DF_input.copy()
    
    # Verificar que existe columna ELEV
    if 'ELEV' not in DF.columns:
        raise ValueError("DataFrame debe tener columna 'ELEV'")
    
    # Extraer alturas
    alturas = DF.ELEV.values.reshape(-1, 1)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters_originales = kmeans.fit_predict(alturas)
    
    # Crear DataFrame temporal para análisis
    temp_df = pd.DataFrame({
        'index': DF.index,
        'ELEV': alturas.flatten(),
        'cluster_orig': clusters_originales
    })
    
    # Calcular estadísticas por cluster original
    stats = temp_df.groupby('cluster_orig').agg({
        'ELEV': ['mean', 'min', 'max', 'count']
    }).round(2)
    
    stats.columns = ['media', 'minimo', 'maximo', 'conteo']
    stats = stats.sort_values('media')  # Ordenar por altura media
    
    # Crear mapeo de reordenamiento
    # Orden ascendente: cluster con menor altura media -> grupo 1
    cluster_mapping = {}
    for nuevo_grupo, cluster_orig in enumerate(stats.index, 1):
        cluster_mapping[cluster_orig] = nuevo_grupo
    
    # Aplicar reordenamiento
    DF['ELEV_KMEANS'] = temp_df['cluster_orig'].map(cluster_mapping).values
    
    # Incluir los grupos en un solo DF
    DF1 = DF[DF.ELEV_KMEANS==1]
    DF2 = DF[DF.ELEV_KMEANS==2]
    DF3 = DF[DF.ELEV_KMEANS==3]
    DF4 = DF[DF.ELEV_KMEANS==4]

    mean_g1 = np.nanmean(DF1.RED)
    mean_g2 = np.nanmean(DF2.RED)
    mean_g3 = np.nanmean(DF3.RED)
    mean_g4 = np.nanmean(DF4.RED)

    DF_ALL = [DF1, DF2, DF3, DF4]
    means = [mean_g1, mean_g2, mean_g3, mean_g4]
    
    return DF_ALL, means

