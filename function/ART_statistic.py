import os
import numpy as np
import pandas as pd
import scipy.stats as stats

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