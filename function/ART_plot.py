import os
import numpy as np
import geopandas as gpd

from scipy import stats

import cartopy
import cartopy.feature as cf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from ART_downscale import compute_beta, epl_fun, str_exp_fun

veneto_dir = os.path.join('/','media','arturo','Arturo','Data','shapes','Europa','Italy','Veneto.geojson')
if os.path.exists(veneto_dir):
    Veneto = gpd.read_file(veneto_dir)
else:
    raise SystemExit(f"File not found: {veneto_dir}")

def plot_neighborhood(box_3h, lon2d, lat2d, bcond, Station_pos, close_pixel, level_name, nameout, save=False):

    box_lon2d, box_lat2d = np.meshgrid(box_3h['lon'].data, box_3h['lat'].data)

    cmap_bin = plt.get_cmap('viridis', 3)  # Use any colormap you like with 2 discrete levels
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap_bin.N)  

    cmap = plt.cm.Spectral_r
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(4,3),dpi=300)
    gs = gridspec.GridSpec(1,1)

    # ========================================================================================
    ax1 = plt.subplot(gs[0, 0], projection = proj)
    ax1.set_extent([10.5, 13.2, 44.7, 46.8], crs=proj)
    ax1.add_feature(cf.COASTLINE,linewidth=0.1, linestyle='-',edgecolor='k')
    ax1.add_geometries(Veneto.geometry, ccrs.PlateCarree(), edgecolor='k', facecolor='None', linewidth=0.6, alpha=0.5)
    a1 = ax1.pcolormesh(lon2d, lat2d, bcond, cmap=cmap_bin, norm=norm)

    ax1.scatter(lon2d, lat2d, s=0.1, c='k', marker='o')
    ax1.scatter(Station_pos[1], Station_pos[0], s=2, c='r', zorder=10, label='Station')
    ax1.scatter(close_pixel[1], close_pixel[0], s=2, c='k', zorder=10, label='Nearest Neighbour')
    ax1.scatter(box_lon2d, box_lat2d, s=0.5, c='b', label='Box pixels')

    plt.legend(fontsize=4, loc=2)

    gl = ax1.gridlines(crs=proj,draw_labels=True,linewidth=0.2,color='gray',alpha=0.5,linestyle='--')
    gl.top_labels = False
    gl.bottom_labels = False
    gl.right_labels = False
    gl.left_labels =False
    gl.xlabel_style = {'size': 4, 'color': 'k'}
    gl.ylabel_style = {'size': 4, 'color': 'k'}

    cbar = plt.colorbar(a1, pad=0.04, fraction=0.0358, ticks=[0, 1])
    cbar.set_ticks([0.25, 0.75])  
    cbar.set_ticklabels(['False', 'True'])
    cbar.ax.tick_params(labelsize=4)

    ax1.set_title('Pixels Neighborhood for Scales Aggregation', loc='left', fontsize=6)
    ax1.set_title(f'{level_name}',loc='right',fontsize=6)

    # ========================================================================================
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.1, wspace=0.12)

    if save == True:
        print(f'Export as: {nameout}')
        ax1.set_facecolor('white')
        fig.patch.set_alpha(0)
        plt.savefig(nameout,transparent = False,bbox_inches ='tight',pad_inches = 0)

def plot_scales_aggregation(WET_MATRIX, xscales_km, tscales, xscale_ref, tscale_ref, name_out, save=False):

    fig = plt.figure(figsize=(6,5),dpi=300)
    gs = gridspec.GridSpec(2,1)

    # ========================================================================================
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(tscales, WET_MATRIX[:,xscale_ref], linewidth=0.7, c='k')
    ax1.plot(tscales, WET_MATRIX[:,xscale_ref], '.', c='k')

    ax1.grid(linewidth=0.3, linestyle='--')
    ax1.set_xlabel('Times Scales [hour]', fontsize=6)
    ax1.set_ylabel('Wet Fraction [dimensionless]', fontsize=6)

    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.yaxis.set_tick_params(labelsize=5)

    ax1.set_xticks(tscales)

    ax1.set_title(f'(a) Wet fraction for different time scales', loc='left', fontsize=8, pad=3)
    ax1.set_title(f'{xscales_km[xscale_ref]} km', loc='right', fontsize=8, pad=3)

    # ========================================================================================

    ax1 = plt.subplot(gs[1, 0])
    ax1.plot(xscales_km, WET_MATRIX[tscale_ref,:], linewidth=0.7, c='k')
    ax1.plot(xscales_km, WET_MATRIX[tscale_ref,:], '.', c='k')

    ax1.grid(linewidth=0.3, linestyle='--')
    ax1.set_xlabel('Spatial Scales [km]', fontsize=6)
    ax1.set_ylabel('Wet Fraction [dimensionless]', fontsize=6)

    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.yaxis.set_tick_params(labelsize=5)

    ax1.set_title(f'(b) Wet fraction for different Spatial scales', loc='left', fontsize=8, pad=3)
    ax1.set_title(f'{tscales[tscale_ref]} hours', loc='right', fontsize=8, pad=3)

    # ========================================================================================
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.3, wspace=0.1)

    ax1.set_facecolor('white')
    fig.patch.set_alpha(0)

    if save == True:
        print(f'Export as: {name_out}')
        plt.savefig(name_out, transparent=False, bbox_inches='tight', pad_inches=0)

def plot_wet_fraction_matrix(WET_MATRIX_INTER, xscales_km, tscales, tscales_INTER, WET_MATRIX_EXTRA, new_spatial_scale, origin, target, station, nameout, save=False):
    xscales_km_2d, tscales_2d = np.meshgrid(xscales_km, tscales)
    levels = None

    fig = plt.figure(figsize=(9,4),dpi=300)
    gs = gridspec.GridSpec(1,2)

    # ============================================================
    ax1 = plt.subplot(gs[0, 0])
    a1 = ax1.contourf(xscales_km, tscales_INTER, WET_MATRIX_INTER, cmap='viridis', levels=levels)

    ax1.scatter(origin[0], origin[1], s=15, c='r', zorder=5, marker='^',label='origin')
    ax1.scatter(target[0], target[1], s=15, marker='s', c='r', zorder=5, label ='target')

    ax1.axvline(x=10, color='k', linestyle='--', linewidth=0.2)

    ax1.plot(xscales_km_2d, tscales_2d, '.k', markersize=3)

    ax1.set_title('(a) Original Wet Fraction', loc='left', fontsize=8)
    ax1.set_title(f'{station}',loc='right',fontsize=8)
    ax1.set_xlabel('Spatial Scale (km)', fontsize=8)
    ax1.set_ylabel('Temporal Scale (hours)', fontsize=8)

    ax1.set_yticks(tscales)
    ax1.xaxis.set_tick_params(labelsize=7)
    ax1.yaxis.set_tick_params(labelsize=7)
    ax1.grid(linewidth=0.1, linestyle='--')

    cbar = plt.colorbar(a1)
    cbar.ax.tick_params(labelsize=7)

    ax1.set_xlim(-2,xscales_km[-1])

    # ============================================================
    ax1 = plt.subplot(gs[0, 1])
    a1 = ax1.contourf(new_spatial_scale, tscales_INTER, WET_MATRIX_EXTRA, cmap='viridis', levels=levels)

    ax1.scatter(origin[0], origin[1], s=15, c='r', zorder=5, marker='^',label='origin')
    ax1.scatter(target[0], target[1], s=15, marker='s', c='r', zorder=5, label ='target')

    ax1.axvline(x=10, color='k', linestyle='--', linewidth=0.2)

    ax1.plot(xscales_km_2d, tscales_2d, '.k', markersize=3)

    ax1.set_title('(b) Extrapolate Method', loc='left', fontsize=8)
    ax1.set_xlabel('Spatial Scale (km)', fontsize=8)
    ax1.set_xlim(-2,xscales_km[-1])

    ax1.set_yticks(tscales)
    ax1.xaxis.set_tick_params(labelsize=7)
    ax1.yaxis.set_tick_params(labelsize=7)
    ax1.grid(linewidth=0.1, linestyle='--')

    cbar = plt.colorbar(a1)
    cbar.ax.tick_params(labelsize=7)

    beta = compute_beta(WET_MATRIX_EXTRA, origin, target, new_spatial_scale, tscales_INTER)
    ax1.set_title(f'beta: {np.round(beta,3)}', loc='right', fontsize=8)

    # ================================================================================================
    ax1.set_facecolor('white')
    fig.patch.set_alpha(0)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.1, wspace=0.1)

    if save == True:
        print(f'Export as: {nameout}')
        plt.savefig(nameout,transparent = False,bbox_inches ='tight',pad_inches = 0, facecolor=None)

def plot_autocorrelation(vdist, vcorr, FIT_eps, FIT_alp, FIT_d0, FIT_mu0, nameout, save=False):
    fig = plt.figure(figsize=(4,3),dpi=300)
    gs = gridspec.GridSpec(1,1)

    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(vdist, vcorr, s=0.8, c='k', label='Correlation')

    ax1.plot(np.sort(vdist), epl_fun(np.sort(vdist), FIT_eps, FIT_alp), '--g', label='Exp. Power law (Marani 2003)')
    ax1.plot(np.sort(vdist), str_exp_fun(np.sort(vdist), FIT_d0, FIT_mu0), '--r',label='Stretched Exp (krajewski 2003)')

    ax1.set_xlim([-5, min(100, 10*(np.round(np.max(vdist))//10+1.5))])
    ax1.set_ylim([0.5, 1.0])

    ax1.grid(linewidth=0.3, linestyle='--')
    ax1.legend(fontsize = 5)

    ax1.xaxis.set_tick_params(labelsize=6)
    ax1.yaxis.set_tick_params(labelsize=6)

    ax1.set_title('Downscaling correlation function', fontsize=8, loc='left')
    ax1.set_xlabel('distance [Km]', fontsize=6)
    ax1.set_ylabel('correlation [-]', fontsize=6)

    # ================================================================================================
    ax1.set_facecolor('white')
    fig.patch.set_alpha(0)

    if save == True:
        print(f'Export as: {nameout}')
        plt.savefig(nameout,transparent = False,bbox_inches ='tight',pad_inches = 0.01, facecolor=None)

def plot_scatter(OBS, IMERG, station_name, nameout, save=False):
    fig = plt.figure(figsize=(4,4),dpi=300)
    gs = gridspec.GridSpec(1,1)
    ax1 = plt.subplot(gs[0, 0])
    mask = ~np.isnan(OBS) & ~np.isnan(IMERG)
    OBS_clear = OBS[mask]
    IMERG_clear = IMERG[mask]

    min_val = np.fmin(np.nanmin(OBS_clear), np.nanmin(IMERG_clear))
    max_val = np.fmax(np.nanmax(OBS_clear), np.nanmax(IMERG_clear))
    x_vals = np.linspace(min_val, max_val, 100)

    slope, intercept, r_value, p_value, std_err = stats.linregress(OBS_clear, IMERG_clear)
    y_vals = slope * x_vals + intercept

    ax1.scatter(OBS_clear, IMERG_clear, s=1, label='Scatter')
    
    ax1.plot(x_vals, y_vals, 'r--', linewidth=0.6, label=f'Linear Regression')
    ax1.plot(x_vals,x_vals,'-', linewidth=0.6, c='k', label='Identity line')

    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.yaxis.set_tick_params(labelsize=5)
    ax1.set_xlim(right=max_val + 10)
    ax1.set_ylim(top=max_val + 10)

    ax1.legend(fontsize=6, ncol=1)
    ax1.grid(linewidth=0.3, linestyle='--')

    ax1.set_xlabel('OBS', fontsize=5)
    ax1.set_ylabel('IMERG', fontsize=5)

    ax1.set_title('Scatter Plot for OBS and IMERG', fontsize=7, loc='left')
    ax1.set_title(f'{station_name}',loc='right',fontsize=7)
    
    ax1.set_facecolor('white')
    fig.patch.set_alpha(0)

    if save == True:
        print(f'Export as: {nameout}')
        plt.savefig(nameout,transparent = False,bbox_inches ='tight',pad_inches = 0.01, facecolor=None)
