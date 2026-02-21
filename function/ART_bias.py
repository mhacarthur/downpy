def linear_bc_through_origin(obs, sat, sat_all):
    
    # DataFrame para limpiar NaN e inf
    df = pd.DataFrame({'OBS': obs, 'SAT': sat})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    x = df['SAT'].values
    y = df['OBS'].values

    # coeficiente forzado al origen
    alpha = np.sum(x * y) / np.sum(x**2)

    # aplicar corrección
    sat_corrected = sat_all * alpha
    # sat_corrected = sat_all / alpha

    return sat_corrected
    