import numpy as np
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