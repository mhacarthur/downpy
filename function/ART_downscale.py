
import numpy as np
from scipy.special import gamma
from scipy.integrate import dblquad, nquad
from scipy.optimize import curve_fit, minimize, fsolve

def wei_fit_update(sample):
    sample = np.asarray(sample) # from list to Numpy array
    x      = np.sort(sample) # sort ascend by default
    M0hat  = np.mean(x)
    M1hat  = 0.0
    n      = x.size # sample size
    for ii in range(n):
        real_ii = ii + 1
        M1hat   = M1hat + x[ii]*(n - real_ii)
    M1hat = M1hat/(n*(n-1))
    c     = M0hat/gamma( np.log(M0hat/M1hat)/np.log(2)) # scale par
    w     = np.log(2)/np.log(M0hat/(2*M1hat)) # shape par
    return  n, c, w

def fit_yearly_weibull_update(xdata, thresh=0, maxmiss=36):
    '''
    This Function computes the Weibull fit parameters for each year in the data.
    Return NCW matrix [WET_DAYS (N), SCALE (S), SHAPE (W), YEARS]
    '''
    OBS_min = 366 - maxmiss
    years = np.unique(xdata.time.dt.year.values)
    years_num = np.size(years)

    NCW = np.zeros((years_num, 4))
    NOBS = np.zeros(years_num)

    for i, yy in enumerate(years):
        sample = xdata.sel(time=str(yy))
        NOBS[i] = len(sample[sample>=0])
        
        if NOBS[i] < OBS_min:
            NCW[i, 0:3] = np.array([0, np.nan, np.nan])
            NCW[i,3] = yy

        else:
            excesses = sample[sample > thresh] - thresh
            Ni = np.size(excesses)
            if Ni == 0:
                NCW[i, 0:3] = np.array([0, np.nan, np.nan])
                NCW[i,3] = yy
            elif Ni == 1:
                NCW[i, 0:3] = np.array([0, np.nan, np.nan])
                NCW[i,3] = yy
            else:
                NCW[i, 0:3] = wei_fit_update(excesses)
                NCW[i,3] = yy

    return NCW

def compute_beta(WET_MATRIX_EXTRA, origin, target, new_spatial_scale, tscales_INTER):
    pos_xmin = np.argmin(np.abs(origin[0] - new_spatial_scale))
    pos_tmin = np.argmin(np.abs(origin[1] - tscales_INTER))
    pwet_origin = WET_MATRIX_EXTRA[pos_tmin, pos_xmin]

    pos_xmin = np.argmin(np.abs(target[0] - new_spatial_scale))
    pos_tmin = np.argmin(np.abs(target[1] - tscales_INTER))
    pwet_target = WET_MATRIX_EXTRA[pos_tmin, pos_xmin]

    beta = pwet_origin / pwet_target
    return beta

def str_exp_fun(x, d0, mu):
    '''
    Stretched exponential rainfall correlation function
    from eg Habib and krajewski (2003)
    or Villarini and Krajewski (2007)
    with 2 parameters d0, mu (value as x = 0 is 1)
    '''
    x = np.asarray(x) # transform to numpy array
    is_scalar = False if x.ndim > 0 else True # create flag for output
    x.shape = (1,)*(1-x.ndim) + x.shape # give it dimension 1 if scalar
    myfun = np.exp( -(x/d0)**mu)
    myfun = myfun if not is_scalar else myfun[0]
    return  myfun

def epl_fun(x, epsilon, alpha):
    '''
    Marco's Exponential kernel + Power law tail
    (autocorrelation) function with exponential nucleus and power law decay
    for x> epsilon - 2 parameters
    see Marani 2003 WRR for more details
    '''
    x = np.asarray(x) # transform to numpy array
    is_scalar = False if x.ndim > 0 else True # create flag for output
    x.shape = (1,)*(1-x.ndim) + x.shape # give it dimension 1 if scalar
    m = np.size(x)
    myfun = np.zeros(m)
    for ii in range(m):
        if x[ii] < 10e-6:
            myfun[ii] = 1
        elif x[ii] < epsilon:
            myfun[ii] = np.exp(-alpha*x[ii]/epsilon)
        else:
            myfun[ii] = (epsilon/(np.exp(1)*x[ii]))**alpha
    myfun = myfun if not is_scalar else myfun[0]
    return  myfun

def str_exp_fun(x, d0, mu):
    '''
    Stretched exponential rainfall correlation function
    from eg Habib and krajewski (2003)
    or Villarini and Krajewski (2007)
    with 2 parameters d0, mu (value as x = 0 is 1)
    '''
    x = np.asarray(x) # transform to numpy array
    is_scalar = False if x.ndim > 0 else True # create flag for output
    x.shape = (1,)*(1-x.ndim) + x.shape # give it dimension 1 if scalar
    myfun = np.exp( -(x/d0)**mu)
    myfun = myfun if not is_scalar else myfun[0]
    return  myfun

def epl_fun(x, epsilon, alpha):
    '''
    Marco's Exponential kernel + Power law tail
    (autocorrelation) function with exponential nucleus and power law decay
    for x> epsilon - 2 parameters
    see Marani 2003 WRR for more details
    '''
    x = np.asarray(x) # transform to numpy array
    is_scalar = False if x.ndim > 0 else True # create flag for output
    x.shape = (1,)*(1-x.ndim) + x.shape # give it dimension 1 if scalar
    m = np.size(x)
    myfun = np.zeros(m)
    for ii in range(m):
        if x[ii] < 10e-6:
            myfun[ii] = 1
        elif x[ii] < epsilon:
            myfun[ii] = np.exp(-alpha*x[ii]/epsilon)
        else:
            myfun[ii] = (epsilon/(np.exp(1)*x[ii]))**alpha
    myfun = myfun if not is_scalar else myfun[0]
    return  myfun

def myfun_sse(xx, yobs, parhat, L, acf):
    xx = np.asarray(xx)
    myacf = lambda x, y, parhat: myacf_2d(x, y, parhat, acf=acf)
    # Ty = np.array([L, 0., L, 0.])
    sse = 0
    m = np.size(xx)
    for ii in range(m):
        myx = xx[ii]
        Tx = np.array([np.abs(L - myx), myx, L + myx, myx])
        # L2 = [L, L]
        # res = corla_2d(L2, L2, parhat, myacf, Tx, Ty, err_min=1e-5)
        # faster option: - same result (optimized)
        res = fast_corla_2d(parhat, myacf, Tx, L, err_min=1e-2)
        sse = sse + (res - yobs[ii]) ** 2
        # sse = sse + ((res - yobs[ii])/yobs[ii]) ** 2 # norm
    # print('sse', sse)
    # print('parhat =', parhat)
    return sse

def myacf_2d(x, y, parhat, acf): #default acf = 'str'
    '''########################################################################
    set of 2D autocorrelation functions
    INPUTS::
        x,y = variables of the ACF
        parhat = set of the two parameters of the ACF
        acf: which ACF. possible choices:
            acf = 'str': stretched exponential ACF,
                WITH PARAMETERS: scale d0 and shape mu
            acf = 'mar': Marani 2003 exponential kernel + power law tail
                WITH PARAMETERS: transition point epsilon and shape alpha
    OUTPUTS::
        value of the ACF at a point
    ########################################################################'''
    d = np.sqrt(x**2 + y**2)
    if acf == 'str': # par d0, mu0
        d0 = parhat[0]
        mu = parhat[1]
        return np.exp( -(d/d0)**mu)
    elif acf == 'mar': # par:: epsilon, alpha
        # print('it actually is a Power Law')
        epsilon = parhat[0]
        alpha   = parhat[1]
        return epl_fun(d, epsilon , alpha)

def fast_corla_2d(par_acf, myacf, Tx, L, err_min=1e-2):
    nab_1 = nabla_2d(par_acf, myacf, L, Tx[0], err_min=err_min)
    nab_2 = nabla_2d(par_acf, myacf, L, Tx[1], err_min=err_min)
    nab_3 = nabla_2d(par_acf, myacf, L, Tx[2], err_min=err_min)
    nab_den = nabla_2d(par_acf, myacf, L, L, err_min=err_min)
    if np.abs(nab_den) < 10e-6: # to avoid infinities
        # print('correcting - inf value')
        nab_den = 10e-6
    covla = 2*(nab_1 -2*nab_2 + nab_3)/(4*nab_den)
    # print('parhat =', par_acf)
    return covla

def nabla_2d(par_acf, myacf, T1, T2, err_min = 1e-2):
    '''########################################################################
    EQUATION 13
    Compute the variance function as in Vanmarcke's book.
    INPUTS::
        par_acf = tuple with the parameters of the autocorr function (ACF)
        myacf = ACF in 1,2,or 3 dimensions, with parameters in par_acf
        T1 = 1st dimension of averaging area
        T2 = 2nd dim of the averaging area'''
    if (T1 == 0) or (T2 == 0):
        print('integration domain is zero')
        return 0.0 # integral is zero in this case.
    else:
        fun_XY = lambda x ,y: (T1 - x ) *(T2 - y ) * myacf(x ,y, par_acf)
        myint, myerr = nquad(fun_XY, [[0. ,T1], [0. ,T2]])
        # if myint != 0.0:
        #     rel_err = myerr /myint
        # else:
        #     rel_err = 0
        # if rel_err > err_min:
        #     print('varfun ERROR --> Numeric Integration scheme does not converge')
        #     print('int rel error = ', rel_err)
        #     sys.exit("aargh! there are errors!") # to stop code execution
        return 4.0 * myint

def vrf(L, L0, par_acf, acf):
    '''-------------------------------------------------------------
    compute the variance reduction factor
    between scales L (large) and L0 (small)
    defined as Var[L]/Var[L0]
    INPUT:
        L [km]
        L0 [Km]
        parhat = tuple with ACF parameters (eg epsilon, alpha)
        acf='mar' type of acf (mar or str available)
    OUTPUT:
        gam, variance reduction factor
        ---------------------------------------------------------'''
    def myacf(x, y, parhat, acf):
        if acf == 'str':  # par d0, mu0
            return str_exp_fun(np.sqrt(x ** 2 + y ** 2), parhat[0], parhat[1])
        elif acf == 'mar':  # par:: epsilon, alpha
            return epl_fun(np.sqrt(x ** 2 + y ** 2), parhat[0], parhat[1])
        else:
            print('vrf ERROR: insert a valid auto correlation function')
    # compute variance reduction factor
    fun_XY = lambda x, y: (L - x) * (L - y) * myacf(x, y, par_acf, acf)
    fun_XY0 = lambda x, y: (L0 - x) * (L0 - y) * myacf(x, y, par_acf, acf)
    # its 2D integral a-la Vanmarcke
    int_XY, abserr   = dblquad(fun_XY,  0.0, L,  lambda x: 0.0, lambda x: L)
    int_XY0, abserr0 = dblquad(fun_XY0, 0.0, L0, lambda x: 0.0, lambda x: L0)
    gam  = 4/L**4*int_XY # between scale L and a point
    # gam = (L0 / L) ** 4 * (int_XY / int_XY0)  # between scales L and L0
    return gam

def down_wei(Ns, Cs, Ws, L, L0, beta, par_acf, acf):
    ''' -----------------------------------------------------------------------
    Downscale Weibull parameters from grid cell scale to a subgrid scale:
    compute the downscaled weibull parameters from a large scale L
    to a smaller scale L0, both in Km

    INPUT ::
        Ns, Cs, Ws (original parmaters at scale L - may be scalars or arrays)
        L [km] large scale
        L0 [km] small scale
        gams = ratio between the mean number of wet days at scales L and L0
        (should be larger than 1)
        par_acf: set of paramaters for the acf (tuple)

    OPTIONAL ARGUMENTS ::
        acf: what acf. default is 'str'. others available:
            'str'  ->  for 2p stretched exonential (Krajewski 2003)
            'mar'  ->  for 2p power law with exp kernel (Marani 2003)

        N_bias:: correction to N if comparing satellite and gauges
        default is equal to zero (no correction)

    OUTPUT::
        Nd, Cd, Wd (downscaled parameters)
        gam = variance reduction function
        fval = function value at the end of numerical minimization
    ---------------------------------------------------------------------'''
    Ns = np.asarray(Ns)  # check if scalar input - should be the same for N,C,W
    Cs = np.asarray(Cs)
    Ws = np.asarray(Ws)
    # the three parameter mush have same shape - I only check one here
    is_scalar = False if Cs.ndim > 0 else True
    Ns.shape = (1,) * (1 - Ns.ndim) + Ns.shape
    Cs.shape = (1,) * (1 - Cs.ndim) + Cs.shape
    Ws.shape = (1,) * (1 - Ws.ndim) + Ws.shape
    m = Cs.shape[0]  # length of parameter arrays = number of blocks=
    gam = vrf(L, L0, par_acf, acf=acf)
    print(f'Gamma value: {gam}')

    # prob wet:: correct satellite N adding the average difference
    pws = np.mean(Ns) / 365.25
    Wd = np.zeros(m)
    Cd = np.zeros(m)
    Nd = np.zeros(m)
    for ii in range(m):
        cs = Cs[ii]
        ws = Ws[ii]
        rhs = (1/(gam*beta)) * (((2*ws*gamma(2 / ws))/((gamma(1/ws))**2)) + (gam-1)*pws)
        wpfun = lambda w: (2*w*gamma(2 / w)/(gamma(1/w))**2) - rhs

        res = fsolve(wpfun, 0.1, full_output=True,xtol=1e-06, maxfev=10000)
        
        Wd[ii] = res[0]
        info = res[1]
        fval = info['fvec']
        if fval > 1e-5:
            print('warning - downscaling function:: '
                    'there is something wrong solving fsolve!')
        Cd[ii] = (beta * Wd[ii]) * (cs / ws) * (gamma(1 / ws) / gamma(1 / Wd[ii]))
        Nd[ii] = int( np.rint( Ns[ii] / beta))

    # If Nd, Cd, Wd are a collection (example, list or array) and not a scalar, 
    # return all collection.
    Nd = Nd if not is_scalar else Nd[0]
    Cd = Cd if not is_scalar else Cd[0]
    Wd = Wd if not is_scalar else Wd[0]

    return Nd, Cd, Wd, gam, fval

def down_wei_beta_alpha(Ns, Cs, Ws, beta, gam):
    Ns = np.asarray(Ns)  # check if scalar input - should be the same for N,C,W
    Cs = np.asarray(Cs)
    Ws = np.asarray(Ws)
    # the three parameter mush have same shape - I only check one here
    is_scalar = False if Cs.ndim > 0 else True
    Ns.shape = (1,) * (1 - Ns.ndim) + Ns.shape
    Cs.shape = (1,) * (1 - Cs.ndim) + Cs.shape
    Ws.shape = (1,) * (1 - Ws.ndim) + Ws.shape
    m = Cs.shape[0]  # length of parameter arrays = number of blocks=

    # prob wet:: correct satellite N adding the average difference
    pws = np.mean(Ns) / 365.25
    Wd = np.zeros(m)
    Cd = np.zeros(m)
    Nd = np.zeros(m)
    for ii in range(m):
        cs = Cs[ii]
        ws = Ws[ii]
        rhs = (1/(gam*beta)) * (((2*ws*gamma(2 / ws))/((gamma(1/ws))**2)) + (gam-1)*pws)
        wpfun = lambda w: (2*w*gamma(2 / w)/(gamma(1/w))**2) - rhs

        res = fsolve(wpfun, 0.1, full_output=True,xtol=1e-06, maxfev=10000)
        
        Wd[ii] = res[0]
        info = res[1]
        fval = info['fvec']
        if fval > 1e-5:
            print('warning - downscaling function:: '
                    'there is something wrong solving fsolve!')
        Cd[ii] = (beta * Wd[ii]) * (cs / ws) * (gamma(1 / ws) / gamma(1 / Wd[ii]))
        Nd[ii] = int( np.rint( Ns[ii] / beta))

    # If Nd, Cd, Wd are a collection (example, list or array) and not a scalar, 
    # return all collection.
    Nd = Nd if not is_scalar else Nd[0]
    Cd = Cd if not is_scalar else Cd[0]
    Wd = Wd if not is_scalar else Wd[0]

    return Nd, Cd, Wd