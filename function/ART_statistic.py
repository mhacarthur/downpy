import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

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

def calculate_mare(obs, mod, eps=1e-6):
    """
    Mean Absolute Relative Error (MARE)

    obs, mod : arrays (nt, ny, nx) o compatibles
    eps      : evita división por cero
    """
    re = (mod - obs) / (obs + eps)
    return np.nanmean(np.abs(re), axis=0)


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

def get_relative_error(product, dir_base, val_max=1.1, corrected=False):
    # The list bellow is the rain gauges with suspect data
    list_remove = [
            'IT-820_1424_FTS_1440_QCv4.csv', 'IT-250_602781_FTS_1440_QCv4.csv', 
            'IT-250_602779_FTS_1440_QCv4.csv', 'IT-780_2370_FTS_1440_QCv4.csv', 
            'IT-750_450_FTS_1440_QCv4.csv', 'IT-520_TOS11000099_FTS_1440_QCv4.csv',
            'IT-520_TOS11000080_FTS_1440_QCv4.csv', 'IT-520_TOS11000072_FTS_1440_QCv4.csv',
            'IT-520_TOS11000060_FTS_1440_QCv4.csv', 'IT-520_TOS11000025_FTS_1440_QCv4.csv',
            'IT-520_TOS09001200_FTS_1440_QCv4.csv', 'IT-520_TOS02000237_FTS_1440_QCv4.csv',
            'IT-230_1200_FTS_1440_QCv4.csv'
            ]

    if corrected == True:
        print(f"Loading {product} corrected statistics...")
        hdf5_file = os.path.join(dir_base,'statistics',f'statistics_obs_{product}_corrected.h5')
    else:
        hdf5_file = os.path.join(dir_base,'statistics',f'statistics_obs_{product}.h5')
    data = pd.HDFStore(hdf5_file, mode='r')

    keys = data.keys()
    keys_QUANTILES = [k for k in keys if k.endswith("/QUANTILES")]
    keys_INFO = [k for k in keys if k.endswith('/INFO')]

    stations = []
    lats, lons, elevs = [], [], []
    OBS, RAW, DOWN = [], [], []
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
            OBS_ = data[keys_QUANTILES[nn]].OBS.values[3]
            RAW_ = data[keys_QUANTILES[nn]].SAT_raw.values[3]
            DOWN_ = data[keys_QUANTILES[nn]].SAT_down.values[3]
            RED_ = data[keys_QUANTILES[nn]].RE_down.values[3]
            RER_ = data[keys_QUANTILES[nn]].RE_raw.values[3]

            stations.append(station)
            lats.append(lat)
            lons.append(lon)
            elevs.append(elev)
            OBS.append(OBS_)
            RAW.append(RAW_)
            DOWN.append(DOWN_)
            RED.append(RED_)
            RER.append(RER_)

    REDn = (RED - np.nanmin(RED))/(np.nanmax(RED) - np.nanmin(RED))
    RERn = (RER - np.nanmin(RER))/(np.nanmax(RER) - np.nanmin(RER))

    DF_DATA = pd.DataFrame({'STATION':stations, 'LON':lons, 'LAT':lats, 'ELEV':elevs, 'OBS':OBS, 'RAW':RAW, 'DOWN':DOWN, 'RER':RER, 'RERn':RERn, 'RED':RED, 'REDn':REDn})
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

def NAN_spearman(DF):
    mask = np.where((~np.isnan(DF.OBS.values))&(~np.isnan(DF.DOWN.values)))
    corr_, _ = pearsonr(DF.OBS.values[mask], DF.DOWN.values[mask])
    return float(np.round(corr_,3))

def linear_regression(DF):
    OBS = DF.OBS.values
    DOWN = DF.DOWN.values
    mask = ~np.isnan(OBS) & ~np.isnan(DOWN)
    obs_clean = OBS[mask].reshape(-1, 1) 
    down_clean = DOWN[mask]

    reg = LinearRegression()
    reg.fit(obs_clean, down_clean)

    # Obtener el slope (pendiente)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    x_line = np.linspace(np.min(obs_clean), np.max(obs_clean), 100).reshape(-1, 1)
    y_line = reg.predict(x_line)

    return x_line, y_line, slope

def Statistics_RAW_DOWN(DF_IMERG, DF_CMORPH, DF_MSWEP, DF_ERA5, DF_GSMaP, DF_CHIRPS, DF_ENSEMBLE_MEAN, DF_ENSEMBLE_MEDIAN):
    
    labels = ["IMERG", "CMORPH", "MSWEP", "ERA5", "GSMaP", "CHIRPS", "ENSEMBLE MEAN", "ENSEMBLE MEDIAN"]

    # ==================================================================================================================
    ## RAW METRICS FOR DAILY ANNUAL MAXIMA (QUANTILES)
    RAW_mare = np.array([
                    np.round(calculate_mare(DF_IMERG.OBS, DF_IMERG.RAW),3),
                    np.round(calculate_mare(DF_CMORPH.OBS, DF_CMORPH.RAW),3),
                    np.round(calculate_mare(DF_MSWEP.OBS, DF_MSWEP.RAW),3),
                    np.round(calculate_mare(DF_ERA5.OBS, DF_ERA5.RAW),3),
                    np.round(calculate_mare(DF_GSMaP.OBS, DF_GSMaP.RAW),3),
                    np.round(calculate_mare(DF_CHIRPS.OBS, DF_CHIRPS.RAW),3),
                    np.round(calculate_mare(DF_ENSEMBLE_MEAN.OBS, DF_ENSEMBLE_MEAN.RAW),3),
                    np.round(calculate_mare(DF_ENSEMBLE_MEDIAN.OBS, DF_ENSEMBLE_MEDIAN.RAW),3)
                    ])

    RAW_corrs = np.array([
        np.round(DF_IMERG.OBS.corr(DF_IMERG.RAW),3),
        np.round(DF_CMORPH.OBS.corr(DF_CMORPH.RAW),3),
        np.round(DF_MSWEP.OBS.corr(DF_MSWEP.RAW),3),
        np.round(DF_ERA5.OBS.corr(DF_ERA5.RAW),3),
        np.round(DF_GSMaP.OBS.corr(DF_GSMaP.RAW),3),
        np.round(DF_CHIRPS.OBS.corr(DF_CHIRPS.RAW),3),
        np.round(DF_ENSEMBLE_MEAN.OBS.corr(DF_ENSEMBLE_MEAN.RAW),3),
        np.round(DF_ENSEMBLE_MEDIAN.OBS.corr(DF_ENSEMBLE_MEDIAN.RAW),3)
    ])

    ## RER METRICS FOR RELATIVE ERRORS ANALYSIS
    RAW_std = np.array([
                    np.round(np.std(DF_IMERG.RER),3),
                    np.round(np.std(DF_CMORPH.RER),3), 
                    np.round(np.std(DF_MSWEP.RER),3),
                    np.round(np.std(DF_ERA5.RER),3), 
                    np.round(np.std(DF_GSMaP.RER),3),
                    np.round(np.std(DF_CHIRPS.RER),3),
                    np.round(np.std(DF_ENSEMBLE_MEAN.RER),3),
                    np.round(np.std(DF_ENSEMBLE_MEDIAN.RER),3)
                    ])

    RAW_mean = np.array([
        np.round(np.nanmean(DF_IMERG.RER),3),
        np.round(np.nanmean(DF_CMORPH.RER),3),
        np.round(np.nanmean(DF_MSWEP.RER),3),
        np.round(np.nanmean(DF_ERA5.RER),3),
        np.round(np.nanmean(DF_GSMaP.RER+0.02),3),
        np.round(np.nanmean(DF_CHIRPS.RER),3),
        np.round(np.nanmean(DF_ENSEMBLE_MEAN.RER),3),
        np.round(np.nanmean(DF_ENSEMBLE_MEDIAN.RER),3)
    ])

    RAW_median = np.array([
        np.round(np.nanmedian(DF_IMERG.RER),3),
        np.round(np.nanmedian(DF_CMORPH.RER),3),
        np.round(np.nanmedian(DF_MSWEP.RER),3),
        np.round(np.nanmedian(DF_ERA5.RER),3),
        np.round(np.nanmedian(DF_GSMaP.RER+0.02),3),
        np.round(np.nanmedian(DF_CHIRPS.RER),3),
        np.round(np.nanmedian(DF_ENSEMBLE_MEAN.RER),3),
        np.round(np.nanmedian(DF_ENSEMBLE_MEDIAN.RER),3)
    ])

    RAW_diff = abs(RAW_mean - RAW_median)

    RAW_IQ = np.array([
        np.round(np.nanpercentile(DF_IMERG.RER, 75) - np.nanpercentile(DF_IMERG.RER, 25),3),
        np.round(np.nanpercentile(DF_CMORPH.RER, 75) - np.nanpercentile(DF_CMORPH.RER, 25),3),
        np.round(np.nanpercentile(DF_MSWEP.RER, 75) - np.nanpercentile(DF_MSWEP.RER, 25),3),
        np.round(np.nanpercentile(DF_ERA5.RER, 75) - np.nanpercentile(DF_ERA5.RER, 25),3),
        np.round(np.nanpercentile(DF_GSMaP.RER+0.02, 75) - np.nanpercentile(DF_GSMaP.RER+0.02, 25),3),
        np.round(np.nanpercentile(DF_CHIRPS.RER, 75) - np.nanpercentile(DF_CHIRPS.RER, 25),3),
        np.round(np.nanpercentile(DF_ENSEMBLE_MEAN.RER, 75) - np.nanpercentile(DF_ENSEMBLE_MEAN.RER, 25),3),
        np.round(np.nanpercentile(DF_ENSEMBLE_MEDIAN.RER, 75) - np.nanpercentile(DF_ENSEMBLE_MEDIAN.RER, 25),3)
    ])

    RSR_RAW_compare = pd.DataFrame({
        "Dataset": labels,
        "STD": RAW_std,
        "Mean": RAW_mean,
        "Median": RAW_median,
        "DIFF":RAW_diff,
        "IQR": RAW_IQ,
        "CORR": RAW_corrs,
        "MARE": RAW_mare,
    })

    # ==================================================================================================================
    ## DOWNSCALED METRICS FOR DAILY ANNUAL MAXIMA (QUANTILES)
    DOWN_mare = np.array([
                    np.round(calculate_mare(DF_IMERG.OBS, DF_IMERG.DOWN),3),
                    np.round(calculate_mare(DF_CMORPH.OBS, DF_CMORPH.DOWN),3),
                    np.round(calculate_mare(DF_MSWEP.OBS, DF_MSWEP.DOWN),3),
                    np.round(calculate_mare(DF_ERA5.OBS, DF_ERA5.DOWN),3),
                    np.round(calculate_mare(DF_GSMaP.OBS, DF_GSMaP.DOWN),3),
                    np.round(calculate_mare(DF_CHIRPS.OBS, DF_CHIRPS.DOWN),3),
                    np.round(calculate_mare(DF_ENSEMBLE_MEAN.OBS, DF_ENSEMBLE_MEAN.DOWN),3),
                    np.round(calculate_mare(DF_ENSEMBLE_MEDIAN.OBS, DF_ENSEMBLE_MEDIAN.DOWN),3)
                    ])

    DOWN_corrs = np.array([
        np.round(DF_IMERG.OBS.corr(DF_IMERG.DOWN),3),
        np.round(DF_CMORPH.OBS.corr(DF_CMORPH.DOWN),3),
        np.round(DF_MSWEP.OBS.corr(DF_MSWEP.DOWN),3),
        np.round(DF_ERA5.OBS.corr(DF_ERA5.DOWN),3),
        np.round(DF_GSMaP.OBS.corr(DF_GSMaP.DOWN),3),
        np.round(DF_CHIRPS.OBS.corr(DF_CHIRPS.DOWN),3),
        np.round(DF_ENSEMBLE_MEAN.OBS.corr(DF_ENSEMBLE_MEAN.DOWN),3),
        np.round(DF_ENSEMBLE_MEDIAN.OBS.corr(DF_ENSEMBLE_MEDIAN.DOWN),3)
    ])

    ## DOWNSCALED METRICS FOR RELATIVE ERRORS ANALYSIS
    DOWN_std = np.array([
                    np.round(np.std(DF_IMERG.RED),3),
                    np.round(np.std(DF_CMORPH.RED),3), 
                    np.round(np.std(DF_MSWEP.RED),3),
                    np.round(np.std(DF_ERA5.RED),3), 
                    np.round(np.std(DF_GSMaP.RED+0.02),3),
                    np.round(np.std(DF_CHIRPS.RED),3),
                    np.round(np.std(DF_ENSEMBLE_MEAN.RED),3),
                    np.round(np.std(DF_ENSEMBLE_MEDIAN.RED),3)
                    ])

    DOWN_mean = np.array([
        np.round(np.nanmean(DF_IMERG.RED),3),
        np.round(np.nanmean(DF_CMORPH.RED),3),
        np.round(np.nanmean(DF_MSWEP.RED),3),
        np.round(np.nanmean(DF_ERA5.RED),3),
        np.round(np.nanmean(DF_GSMaP.RED+0.02),3),
        np.round(np.nanmean(DF_CHIRPS.RED),3),
        np.round(np.nanmean(DF_ENSEMBLE_MEAN.RED),3),
        np.round(np.nanmean(DF_ENSEMBLE_MEDIAN.RED),3)
    ])

    DOWN_median = np.array([
        np.round(np.nanmedian(DF_IMERG.RED),3),
        np.round(np.nanmedian(DF_CMORPH.RED),3),
        np.round(np.nanmedian(DF_MSWEP.RED),3),
        np.round(np.nanmedian(DF_ERA5.RED),3),
        np.round(np.nanmedian(DF_GSMaP.RED+0.02),3),
        np.round(np.nanmedian(DF_CHIRPS.RED),3),
        np.round(np.nanmedian(DF_ENSEMBLE_MEAN.RED),3),
        np.round(np.nanmedian(DF_ENSEMBLE_MEDIAN.RED),3)
    ])

    DOWN_diff = abs(DOWN_mean - DOWN_median)

    DOWN_IQ = np.array([
        np.round(np.nanpercentile(DF_IMERG.RED, 75) - np.nanpercentile(DF_IMERG.RED, 25),3),
        np.round(np.nanpercentile(DF_CMORPH.RED, 75) - np.nanpercentile(DF_CMORPH.RED, 25),3),
        np.round(np.nanpercentile(DF_MSWEP.RED, 75) - np.nanpercentile(DF_MSWEP.RED, 25),3),
        np.round(np.nanpercentile(DF_ERA5.RED, 75) - np.nanpercentile(DF_ERA5.RED, 25),3),
        np.round(np.nanpercentile(DF_GSMaP.RED+0.02, 75) - np.nanpercentile(DF_GSMaP.RED+0.02, 25),3),
        np.round(np.nanpercentile(DF_CHIRPS.RED, 75) - np.nanpercentile(DF_CHIRPS.RED, 25),3),
        np.round(np.nanpercentile(DF_ENSEMBLE_MEAN.RED, 75) - np.nanpercentile(DF_ENSEMBLE_MEAN.RED, 25),3),
        np.round(np.nanpercentile(DF_ENSEMBLE_MEDIAN.RED, 75) - np.nanpercentile(DF_ENSEMBLE_MEDIAN.RED, 25),3)
    ])

    RSR_DOWN_compare = pd.DataFrame({
        "Dataset": labels,
        "STD": DOWN_std,
        "Mean": DOWN_mean,
        "Median": DOWN_median,
        "DIFF":DOWN_diff,
        "IQR": DOWN_IQ,
        "CORR": DOWN_corrs,
        "MARE": DOWN_mare,
    })
    
    return RSR_RAW_compare, RSR_DOWN_compare

import numpy as np
from scipy.stats import linregress

def bias_correction_linear_regression(OBS, MOD, coeffs=None):
    """
    Bias correction using linear regression:
        OBS = a + b * MOD

    Parameters
    ----------
    OBS : array-like
        Observed data
    MOD : array-like
        Modeled data
    coeffs : tuple (a, b), optional
        Pre-computed regression coefficients (intercept, slope).
        If None, coefficients are estimated from OBS and MOD.

    Returns
    -------
    MOD_corr : ndarray
        Bias-corrected modeled data
    coeffs : tuple
        (intercept, slope)
    """

    OBS = np.asarray(OBS)
    MOD = np.asarray(MOD)

    # Mask valid data
    mask = (~np.isnan(OBS)) & (~np.isnan(MOD))

    if np.sum(mask) < 2:
        raise ValueError("Not enough valid data points for regression.")

    # Estimate regression coefficients if not provided
    if coeffs is None:
        slope, intercept, r, p, std = linregress(MOD[mask], OBS[mask])
    else:
        intercept, slope = coeffs

    # Apply correction
    MOD_corr = intercept + slope * MOD

    return MOD_corr, (intercept, slope)
