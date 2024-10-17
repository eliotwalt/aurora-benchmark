import os
import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import math

from aurora_benchmark.utils import Statistics

np.random.seed(0)

sns.set_theme(
    style="whitegrid",
)

def rmse_curves(
    global_statistics: Statistics,
    med_statistics: Statistics,
    med_wet_statistics: Statistics,
    med_dry_statistics: Statistics,
    fig_title: str,
    std_fig_title: str,
    eval_dir: str,
    nrows: int=4,
    variables: list[str]=None,
    std_plot: bool=False,
):
    if variables is None or len(variables) == 0:
        all_vars = global_statistics["surface_vars"].variables
        for var in global_statistics["atmospheric_vars"].variables:
            for level in global_statistics["atmospheric_vars"].means[var].level.values:
                all_vars.append(f"{var}_{int(level)}")
    else:
        all_vars = variables
        
    fig_curves, axes_curves = plt.subplots(
        ncols=len(all_vars)//nrows,
        nrows=nrows, 
        figsize=(15, 10)
    )
    if std_plot:
        fig_curves_stds, axes_curves_stds = plt.subplots(
            ncols=len(all_vars)//nrows,
            nrows=nrows, 
            figsize=(15, 10)
        )

    for i, var in enumerate(all_vars):
        # get key
        variable_name = var
        if var in global_statistics["surface_vars"].variables:
            stat_key = "surface_vars"
        else:
            stat_key = "atmospheric_vars"
            var, level = var.split("_")
            level = float(level)
        # compute rmses
        global_rmse = global_statistics[stat_key].rmse(dim=["time", "latitude", "longitude"])[var]
        med_rmse = med_statistics[stat_key].rmse(dim=["time", "latitude", "longitude"])[var]
        med_wet_rmse = med_wet_statistics[stat_key].rmse(dim=["time", "latitude", "longitude"])[var]
        med_dry_rmse = med_dry_statistics[stat_key].rmse(dim=["time", "latitude", "longitude"])[var]
        if stat_key == "atmospheric_vars":
            global_rmse = global_rmse.sel(level=level)
            med_rmse = med_rmse.sel(level=level)
            med_wet_rmse = med_wet_rmse.sel(level=level)
            med_dry_rmse = med_dry_rmse.sel(level=level)
            
        # compute rmses stds (if necessary)
        if std_plot:
            global_rmse_std = global_statistics[stat_key].rmse(dim=["time", "latitude", "longitude"], reduce="std")[var]
            med_rmse_std = med_statistics[stat_key].rmse(dim=["time", "latitude", "longitude"], reduce="std")[var]
            med_wet_rmse_std = med_wet_statistics[stat_key].rmse(dim=["time", "latitude", "longitude"], reduce="std")[var]
            med_dry_rmse_std = med_dry_statistics[stat_key].rmse(dim=["time", "latitude", "longitude"], reduce="std")[var]
            if stat_key == "atmospheric_vars":
                global_rmse_std = global_rmse_std.sel(level=level)
                med_rmse_std = med_rmse_std.sel(level=level)
                med_wet_rmse_std = med_wet_rmse_std.sel(level=level)
                med_dry_rmse_std = med_dry_rmse_std.sel(level=level)
            
        # plot all regions
        for rmse, rmse_std, label in zip(
            [global_rmse, med_rmse, med_wet_rmse, med_dry_rmse],
            [global_rmse_std, med_rmse_std, med_wet_rmse_std, med_dry_rmse_std],
            ["Global", "MED", "MED wet season", "MED dry season"]
        ):
            lead_times = [pd.Timedelta(lt)/pd.Timedelta("1w") for lt in rmse.lead_time.values]
            # without stds
            axc = axes_curves.flat[i]
            axc.plot(
                lead_times,
                rmse,
                label=label
            )
            axc.set_title(f"{variable_name}")
            if i % (len(all_vars) // nrows) == 0:
                axc.set_ylabel("RMSE")
            if i >= len(all_vars) - (len(all_vars) // nrows):
                axc.set_xlabel("Lead time (weeks)")
            axc.grid(False)
            # with stds
            if std_plot:
                axs = axes_curves_stds.flat[i]
                axs.plot(
                    lead_times,
                    rmse,
                    label=label
                )
                axs.fill_between(
                    lead_times,
                    rmse - rmse_std,
                    rmse + rmse_std,
                    alpha=0.2
                )
                axs.set_title(f"{variable_name}")
                if i % (len(all_vars) // nrows) == 0:
                    axs.set_ylabel("RMSE")
                if i >= len(all_vars) - (len(all_vars) // nrows):
                    axs.set_xlabel("Lead time (weeks)")
                axs.grid(False)

        # Add legend at the bottom of the rmse figure
        handles, labels = axc.get_legend_handles_labels()
        fig_curves.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
        # Add title
        fig_curves.suptitle(fig_title)#f"RMSE for base_frequency={base_frequency}, eval_aggregation={eval_aggregation}, eval_start={eval_start}, forecast_horizon={forecast_horizon}\n")
        
        if std_plot:
            # Add legend at the bottom of the rmse+stds figure
            handles, labels = axs.get_legend_handles_labels()
            fig_curves_stds.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
            # Add title
            fig_curves_stds.suptitle(std_fig_title)#f"RMSE for base_frequency={base_frequency}, eval_aggregation={eval_aggregation}, eval_start={eval_start}, forecast_horizon={forecast_horizon}\n")

    fig_curves.tight_layout()
    fig_curves.savefig(f"{eval_dir}/rmse_curves.png")
    fig_curves.show()

    if std_plot:
        fig_curves_stds.tight_layout()
        fig_curves_stds.savefig(f"{eval_dir}/rmse_curves_stds.png", dpi=300, bbox_inches='tight')
        fig_curves_stds.show()

def signed_difference_maps(
    global_statistics: Statistics,
    med_statistics: Statistics,
    med_wet_statistics: Statistics,
    med_dry_statistics: Statistics,
    eval_dir: str,
    variables: list[str]=None,
):
    if variables is None or len(variables) == 0:
        all_vars = global_statistics["surface_vars"].variables
        for var in global_statistics["atmospheric_vars"].variables:
            for level in global_statistics["atmospheric_vars"].means[var].level.values:
                all_vars.append(f"{var}_{int(level)}")
    else:
        all_vars = variables

    for i, var in enumerate(all_vars):
        # get key
        variable_name = var
        if var in global_statistics["surface_vars"].variables:
            stat_key = "surface_vars"
        else:
            stat_key = "atmospheric_vars"
            var, level = var.split("_")
            level = float(level)
        
        # get the mean signed differences
        global_diff = global_statistics[stat_key].means[var]
        med_diff = med_statistics[stat_key].means[var]
        med_wet_diff = med_wet_statistics[stat_key].means[var]
        med_dry_diff = med_dry_statistics[stat_key].means[var]
        if stat_key == "atmospheric_vars":
            global_diff = global_diff.sel(level=level)
            med_diff = med_diff.sel(level=level)
            med_wet_diff = med_wet_diff.sel(level=level)
            med_dry_diff = med_dry_diff.sel(level=level)
        
        # get lead times
        lead_times = global_diff.lead_time.values
        
        # go over "regions"
        for (label, diff) in zip(["global", "MED", "MED wet season", "MED dry season"], [global_diff, med_diff, med_wet_diff, med_dry_diff]):
            
            nrows = 3
            
            # initialise figure
            fig_diffs, axes_diffs = plt.subplots(ncols=math.ceil(len(lead_times)/nrows), nrows=nrows, figsize=(4*nrows,2.5*nrows), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Compute the minimum and maximum of all diffs
            v = np.max([np.abs(diff.max()), np.abs(diff.min())])
            longitudes = np.linspace(diff.longitude.min().item(), diff.longitude.max().item(), 5)
            latitudes = np.linspace(diff.latitude.min().item(), diff.latitude.max().item(), 5)   
            
            imgs = []
            for i, lead_time in enumerate(lead_times):
                ax = axes_diffs.flat[i]
                ax.set_extent([diff.longitude.min().item(), diff.longitude.max().item(), diff.latitude.min().item(), diff.latitude.max().item()], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':') 
                gl = ax.gridlines(draw_labels=False, rotate_labels=90)  # Disable gridline labels
                gl.xlabels_top = False
                gl.ylabels_right = False
                gl.xlocator = mticker.FixedLocator(longitudes)
                gl.ylocator = mticker.FixedLocator(latitudes)
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                # Remove ticks and labels|
                # Plot the first variable in the dataset
                d = diff.sel(lead_time=lead_time)
                imgs.append(d.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap='coolwarm',
                    vmin=-v,
                    vmax=v,
                    add_colorbar=False  # Remove individual colorbars
                ))
                
                lt = int(pd.Timedelta(lead_time) / pd.Timedelta("1w"))
                ax.set_title(f"Lead time: {lt} week(s)")
                
                ax.set_xticks(longitudes, crs=ccrs.PlateCarree())
                ax.set_yticks(latitudes, crs=ccrs.PlateCarree())
                
                if i % (len(lead_times) // nrows) == 0:
                    ax.set_ylabel("Latitude")
                else: 
                    ax.set_ylabel("")
                if i >= len(lead_times) - len(lead_times) // nrows:
                    ax.set_xlabel("Longitude")
                else:
                    ax.set_xlabel("")            
                                
            # remove empty axes
            for i in range(len(lead_times), len(axes_diffs.flat)):
                axes_diffs.flat[i].axis("off")
            
            # Add a single colorbar
            cbar = fig_diffs.colorbar(imgs[0], ax=axes_diffs, orientation='horizontal', extend='both', shrink=0.5,
                                    anchor=(0.5, -1.8), aspect=50)
            
            fig_diffs.suptitle(f"Signed differences for {variable_name} ({label})")
            plt.tight_layout()
            plt.savefig(f"{eval_dir}/signed_differences_{variable_name}_{label}.png", dpi=300, bbox_inches='tight')
            plt.show()
            
def find_closest_files(
    forecast_dir: str,
    target_time: pd.Timestamp,
    variables: list[str]=None
):
    files = list(sorted(os.scandir(forecast_dir), key=lambda x: x.name))
    min_diff = None
    closest_files = None
    closest_indexes = None
    for i, file in enumerate(files):
        file_info = file.name.replace(".nc", "").split("_")
        variable_name = file_info[1]
        if variables is not None and variable_name not in variables:
            continue
        file_info = file_info[2].split("-")
        init_time = pd.Timestamp(file_info[0])
        diff = abs(init_time - target_time)
        if min_diff is None or diff < min_diff:
            min_diff = diff
            closest_files = [file]
            closest_indexes = [i]
        elif diff == min_diff:
            closest_files.append(file)
            closest_indexes.append(i)
    return closest_indexes, closest_files
            
def get_matching_datasets(
    path: str,
    atmospheric_ds: xr.Dataset,
    surface_ds: xr.Dataset,
    variable_name: str,
    init_time: pd.Timestamp,
    eval_aggregation: str,
    eval_start: str,
    forecast_horizon: str,
):
    pred_trajectory = xr.open_dataset(path, engine="netcdf4")
    assert pd.Timedelta((pred_trajectory.lead_time[1]-pred_trajectory.lead_time[0]).values) == pd.Timedelta(eval_aggregation)

    # load ERA5 gt from surface_ds and atmospheric_ds
    true_ds = atmospheric_ds if variable_name in atmospheric_ds.data_vars else surface_ds
    true_trajectory = true_ds[variable_name]\
            .sel(time=slice(init_time+pd.Timedelta(eval_start), init_time+pd.Timedelta(forecast_horizon)))

    # resample gt to eval_aggregation
    true_trajectory = true_trajectory.resample(time=pd.Timedelta(eval_aggregation), origin=init_time).mean()
    assert pd.Timedelta((true_trajectory.time[1]-true_trajectory.time[0]).values) == pd.Timedelta(eval_aggregation)

    # rename true time to lead time
    true_trajectory = true_trajectory.rename({"time": "lead_time"})
    true_trajectory["lead_time"] = true_trajectory["lead_time"] - np.datetime64(init_time)

    # shape
    sizes = pred_trajectory.sizes
    nlt = len(np.unique(pred_trajectory.lead_time.values))
    if variable_name in atmospheric_ds.data_vars:
        stat_key = "atmospheric_vars"
    else:
        stat_key = "surface_vars"
        
    # compute signed error
    error = (pred_trajectory - true_trajectory)
    
    # ensure [-180, 180] longitude 
    if true_trajectory.longitude.max() > 180:
        true_trajectory = true_trajectory.assign_coords({"longitude": true_trajectory.longitude.values-180})
    if pred_trajectory.longitude.max() > 180:
        pred_trajectory = pred_trajectory.assign_coords({"longitude": pred_trajectory.longitude.values-180})
    if error.longitude.max() > 180:
        error = error.assign_coords({"longitude": error.longitude.values-180})
        
    # flip latitude of true and and errpr
    if true_trajectory.latitude.values[0] < true_trajectory.latitude.values[1]: # i.e. if latitude is increasing
        true_trajectory = true_trajectory.assign_coords({"latitude": true_trajectory.latitude.values[::-1]})
    if error.latitude.values[0] < error.latitude.values[1]: # i.e. if latitude is increasing
        error = error.assign_coords({"latitude": error.latitude.values[::-1]})
    if pred_trajectory.latitude.values[0] < pred_trajectory.latitude.values[1]: # i.e. if latitude is increasing
        pred_trajectory = pred_trajectory.assign_coords({"latitude": pred_trajectory.latitude.values[::-1]})

    return pred_trajectory, true_trajectory, error
            
def prediction_maps(
    forecast_dir: str,
    atmospheric_ds: xr.Dataset,
    surface_ds: xr.Dataset,
    eval_dir: str,
    file_index: int=None,
    lead_times: list[pd.Timedelta]=None,
    level: int=None
):

    # define med
    med_region = {    
        "latitude": slice(47, 29), 
        "longitude": slice(-8, 38) 
    }
    
    # get file list
    files = list(sorted(os.scandir(forecast_dir), key=lambda x: x.name))
    if file_index is None:
        file_index = np.random.randint(0, len(files))
    file = files[file_index]
    
    # get info
    file_info = file.name.replace(".nc", "").split("_")
    variable_name = file_info[1]
    file_info = file_info[2].split("-")
    init_time = pd.Timestamp(file_info[0])
    base_frequency = file_info[1]
    eval_aggregation = file_info[2]
    eval_start = file_info[3]
    forecast_horizon = file_info[4] 
    
    print(f"Plotting {file.name}")
    
    # open prediction file
    pred_trajectory, true_trajectory, signed_error_ds = get_matching_datasets(
        file.path, atmospheric_ds, surface_ds,
        variable_name, init_time, eval_aggregation, eval_start, forecast_horizon    
    )
    if lead_times is not None:
        pred_trajectory = pred_trajectory.sel(lead_time=lead_times)
        true_trajectory = true_trajectory.sel(lead_time=lead_times)
        signed_error_ds = signed_error_ds.sel(lead_time=lead_times)
    
    # get med data
    pred_trajectory_med = pred_trajectory.sel(med_region)
    true_trajectory_med = true_trajectory.sel(med_region)
    signed_error_ds_med = signed_error_ds.sel(med_region)
    
    if variable_name in atmospheric_ds.data_vars:
        atmospheric = True
        assert level is not None 
        variable_name = f"{variable_name}_{level}"
    else:
        atmospheric = False
    
    for (pred, true, error), region in [
        ([pred_trajectory, true_trajectory, signed_error_ds], "global"),
        ([pred_trajectory_med, true_trajectory_med, signed_error_ds_med], "MED"),
    ]:        
        fig, axs = plt.subplots(ncols=len(pred.lead_time.values)+1, 
                                nrows=3, 
                                figsize=(2.5*len(pred.lead_time.values), 5),
                                gridspec_kw={"width_ratios": [1 for _ in pred.lead_time.values] + [0.05]},
                                subplot_kw={'projection': ccrs.PlateCarree()})

        var = list(pred.data_vars)[0]

        vmin = np.min([pred[var].min(), true.min()])
        vmax = np.max([pred[var].max(), true.max()])

        v = np.max([np.abs(error[var].max()), np.abs(error[var].min())])

        pred_imgs = []
        true_imgs = []
        diff_imgs = []

        if true.longitude.max() > 180:
            true = true.assign_coords({"longitude": true.longitude.values-180})
            pred = pred.assign_coords({"longitude": pred.longitude.values-180})
            error = error.assign_coords({"longitude": error.longitude.values-180})
            
        bounds = [true.longitude.min().item(), true.longitude.max().item(), true.latitude.min().item(), true.latitude.max().item()]
        longitudes = np.linspace(math.floor(bounds[0]), math.ceil(bounds[1]), 3)
        latitudes = np.linspace(math.floor(bounds[2]), math.ceil(bounds[3]), 3)

        for i in range(axs.shape[1]-1):
            lead_time = pred.lead_time.values[i]
            
            # prediction
            ax = axs[0, i]
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':') 
            gl = ax.gridlines(draw_labels=False, rotate_labels=90)  # Disable gridline labels
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator(longitudes)
            gl.ylocator = mticker.FixedLocator(latitudes)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            # Remove ticks and labels|
            # Plot the first variable in the dataset
            d = pred.sel(lead_time=lead_time)[var]
            if atmospheric:
                d = d.sel(level=level)
            pred_imgs.append(d.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False
            ))
            lt = int(pd.Timedelta(lead_time) / pd.Timedelta("1w"))
            ax.set_title(f"Lead time: {lt} week(s)")
            ax.set_xlabel("")
            ax.set_xticks([], crs=ccrs.PlateCarree())
            if i==0: 
                ax.set_ylabel("latitude")
                ax.set_yticks(latitudes, crs=ccrs.PlateCarree())
            else: 
                ax.set_ylabel("")
                ax.set_yticks([], crs=ccrs.PlateCarree())
                
            # ground truth
            ax = axs[1, i]
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':') 
            gl = ax.gridlines(draw_labels=False, rotate_labels=90)  # Disable gridline labels
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator(longitudes)
            gl.ylocator = mticker.FixedLocator(latitudes)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            # Remove ticks and labels|
            # Plot the first variable in the dataset
            d = true.sel(lead_time=lead_time)
            if atmospheric:
                d = d.sel(level=level)
            true_imgs.append(d.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False
            ))
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_xticks([], crs=ccrs.PlateCarree())
            if i==0: 
                ax.set_ylabel("latitude")
                ax.set_yticks(latitudes, crs=ccrs.PlateCarree())
            else: 
                ax.set_ylabel("")
                ax.set_yticks([], crs=ccrs.PlateCarree())
                
            # differences
            ax = axs[2, i]
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':') 
            gl = ax.gridlines(draw_labels=False, rotate_labels=90)  # Disable gridline labels
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator(longitudes)
            gl.ylocator = mticker.FixedLocator(latitudes)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            # Remove ticks and labels|
            # Plot the first variable in the dataset
            d = error.sel(lead_time=lead_time)[var]
            if atmospheric:
                d = d.sel(level=level)
            diff_imgs.append(d.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap='coolwarm',
                vmin=-v,
                vmax=v,
                add_colorbar=False # Remove individual colorbars
            ))
            ax.set_title("")
            ax.set_xlabel("longitude")
            ax.set_xticks(longitudes, crs=ccrs.PlateCarree())
            if i==0: 
                ax.set_ylabel("latitude")
                ax.set_yticks(latitudes, crs=ccrs.PlateCarree())
            else: 
                ax.set_ylabel("")
                ax.set_yticks([], crs=ccrs.PlateCarree())
                
        fig.colorbar(pred_imgs[-1], ax=axs[0, -1], orientation='vertical', extend="both", fraction=.8, shrink=.8)
        fig.colorbar(true_imgs[-1], ax=axs[1, -1], orientation='vertical', extend="both", fraction=.8, shrink=.8)
        fig.colorbar(diff_imgs[-1], ax=axs[2, -1], orientation='vertical', extend="both", fraction=.8, shrink=.8)
                
        fig.suptitle(f"Prediction, ground truth and signed error for {variable_name} on {init_time} ({region})\n(base_frequency={base_frequency}, eval_aggregation={eval_aggregation}, eval_start={eval_start}, forecast_horizon={forecast_horizon})")    
        fig.tight_layout()
        fig.savefig(f"{eval_dir}/prediction_maps_{variable_name}_{region}_{init_time.strftime('%Y%m%dT%H%M%S')}.png", dpi=300, bbox_inches='tight')
        fig.show()
