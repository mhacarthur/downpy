import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def normalize_series(series):
    min_val = series.min()
    max_val = series.max()
    
    output = 2 * (series - min_val) / (max_val - min_val) - 1

    return output

def test(series):
    min_val = series.min()
    max_val = series.max()
    
    output = 2 * (series - min_val) / (max_val - min_val) - 1

    return output

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

def relative_error(PRE, OBS):
    OBS = np.array(OBS)
    PRE = np.array(PRE)
    mask = (OBS != 0) & (PRE != 0)
    RE = (PRE[mask] - OBS[mask]) / OBS[mask]
    return RE

def spearman_corr(pre, obs):
    '''
    Spearman Rank Correlation Coefficient.
    '''
    # Elimina pares donde al menos uno es NaN
    obs = np.array(obs)
    pre = np.array(pre)
    mask = (obs != 0) & (pre != 0)
    if np.sum(mask) < 2:
        return np.nan  # No se puede calcular con menos de 2 pares válidos

    corr, _ = spearmanr(pre[mask], obs[mask])
    return corr

# ======================================================================================================
# MILTON'S CODE
# Font: https://github.com/SInApSE-INPE/rain_gauge_validation/blob/main/metrics.py

def NMAE(pre,obs):
    '''
    Normalized Mean Absolute Error (nMAE).
    '''
    madx = np.mean(np.abs(obs - np.mean(obs)))  # Desvio absoluto médio de x
    mady = np.mean(np.abs(pre - np.mean(pre)))  # Desvio absoluto médio de y
    mae = np.mean(np.abs(obs - pre))  # Cálculo do MAE
    mean_diff = np.abs(np.mean(obs) - np.mean(pre))  # Diferença absoluta entre as médias
    mae_max = mean_diff + madx + mady  # Cálculo do MAE máximo
    mae_star = mae / mae_max if mae_max != 0 else 0  # Evita divisão por zero
    return mae_star

def NMSE(pre,obs):
    """
    Normalized Mean Squared Error (RMSE).
    """
    d = pre - obs  # Erro de previsão
    d2_mean = np.mean(d ** 2)  # Média dos erros quadráticos
    Sx = np.std(obs, ddof=0)  # Desvio-padrão de x (populacional)
    Sy = np.std(pre, ddof=0)  # Desvio-padrão de y (populacional)
    mse_star = d2_mean / (d2_mean + (Sx + Sy) ** 2)
    return mse_star

def PSS(pre,obs):
    """
    Probability Density Function (PDF) skill score.
    """
    bins=np.arange(0, 100, .1)
    pdf_obs = np.histogram(obs, bins=bins, density=False)[0]/len(obs)
    pdf_pre = np.histogram(pre, bins=bins, density=False)[0]/len(pre)
    return np.sum(np.min([pdf_obs ,pdf_pre],axis=0))

# ======================================================================================================

def MAE(pre,obs):
    '''
    Mean Absolute Error (MAE).
    '''
    mae = np.mean(np.abs(pre - obs))  # Cálculo do MAE
    return mae

def MSE(pre,obs):
    """
    Mean Squared Error (MSE).
    """
    mse = np.mean((pre - obs) ** 2)
    return mse

def RMSE(pred, obs):
    return np.sqrt(np.mean((pred - obs) ** 2))

def MBE(pred, obs):
    return np.mean(pred - obs)

def NSE(sim, obs):
    """
    Calcula el coeficiente de eficiencia de Nash-Sutcliffe (NSE)
    
    Parámetros:
        obs : array-like
            Valores observados
        sim : array-like
            Valores simulados o predichos
        
    Retorna:
        float : NSE (1 = predicción perfecta, 0 = igual que la media observada, <0 = peor)
    """
    obs = np.array(obs)
    sim = np.array(sim)
    
    return 1 - np.sum((sim - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

