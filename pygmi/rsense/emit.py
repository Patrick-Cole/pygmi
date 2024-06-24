# -----------------------------------------------------------------------------
# Name:        emit.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2024 Council for Geoscience
# Licence:     GPL-3.0
#
# This file is part of PyGMI
#
# PyGMI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyGMI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
"""
EMIT is used to import EMIT satellite data into PyGMI.

It uses code by Erik Bolch, ebolch@contractor.usgs.gov
"""

import os
import datetime

import numpy as np
import xarray as xr
from pyproj.crs import CRS

from pygmi.raster.datatypes import Data

# needs xarray, h5netcdf, rioxarray


def emit_xarray(filepath, ortho=False, qmask=None, unpacked_bmask=None):
    """
    EMIT xarray.

    This function utilizes other functions in this module to streamline
    opening an EMIT dataset as an xarray.Dataset.

    Parameters
    ----------
    filepath : str
        a file path to an EMIT netCDF file.
    ortho : bool, optional
        Whether to orthorectify the dataset or leave in crosstrack/downtrack
        coordinates. The default is False.
    qmask : numpy array, optional
        Output from the quality_mask function used to mask
        pixels based on quality flags selected in that function. Any
        non-orthorectified array with the proper crosstrack and downtrack
        dimensions can also be used. The default is None.
    unpacked_bmask : numpy array, optional
        From the band_mask function, used to mask band-specific pixels that
        have been interpolated. The default is None.

    Returns
    -------
    out_xr : xarray.Dataset
        Dataset constructed based on the parameters provided.

    """
    # Grab granule filename to check product
    granule_id = os.path.splitext(os.path.basename(filepath))[0]

    # Read in Data as Xarray Datasets
    engine, wvl_group = "h5netcdf", None

    ds = xr.open_dataset(filepath, engine=engine)
    loc = xr.open_dataset(filepath, engine=engine, group="location")

    # Check if mineral dataset and read in groups (only ds/loc for minunc)

    if "L2B_MIN_" in granule_id:
        wvl_group = "mineral_metadata"
    elif "L2B_MINUNC" not in granule_id:
        wvl_group = "sensor_band_parameters"

    wvl = None

    if wvl_group:
        wvl = xr.open_dataset(filepath, engine=engine, group=wvl_group)

    # Building Flat Dataset from Components
    data_vars = {**ds.variables}

    # Format xarray coordinates based upon emit product (no wvl for mineral
    # uncertainty)
    coords = {
        "downtrack": (["downtrack"], ds.downtrack.data),
        "crosstrack": (["crosstrack"], ds.crosstrack.data),
        **loc.variables,
    }

    product_band_map = {
        "L2B_MIN_": "name",
        "L2A_MASK_": "mask_bands",
        "L1B_OBS_": "observation_bands",
        "L2A_RFL_": "wavelengths",
        "L1B_RAD_": "wavelengths",
        "L2A_RFLUNCERT_": "wavelengths",
    }

    if wvl:
        coords = {**coords, **wvl.variables}

    out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
    out_xr.attrs["granule_id"] = granule_id

    if band := product_band_map.get(
        next((k for k in product_band_map.keys() if k in granule_id),
             "unknown"), None):
        if "minerals" in list(out_xr.dims):
            out_xr = out_xr.swap_dims({"minerals": band})
            out_xr = out_xr.rename({band: "mineral_name"})
        else:
            out_xr = out_xr.swap_dims({"bands": band})

    # Apply Quality and Band Masks, set fill values to NaN
    for var in list(ds.data_vars):
        if qmask is not None:
            out_xr[var].data[qmask == 1] = np.nan
        if unpacked_bmask is not None:
            out_xr[var].data[unpacked_bmask == 1] = np.nan
        out_xr[var].data[out_xr[var].data == -9999] = np.nan

    if ortho is True:
        out_xr = ortho_xr(out_xr)
        out_xr.attrs["Orthorectified"] = "True"

    return out_xr


def apply_glt(ds_array, glt_array, fill_value=-9999, GLT_NODATA_VALUE=0):
    """
    Apply GLT.

    This function applies the GLT array to a numpy array of either 2 or 3
    dimensions.

    Parameters
    ----------
    ds_array : numpy array
        A numpy array of the desired variable.
    glt_array : GLT array
        A GLT array constructed from EMIT GLT data.
    fill_value : int, optional
        Fill value. The default is -9999.
    GLT_NODATA_VALUE : int, optional
        GLT nodata value. The default is 0.

    Returns
    -------
    out_ds : numpy array
        a numpy array of orthorectified data.

    """
    # Build Output Dataset
    if ds_array.ndim == 2:
        ds_array = ds_array[:, :, np.newaxis]
    out_ds = np.full(
        (glt_array.shape[0], glt_array.shape[1], ds_array.shape[-1]),
        fill_value,
        dtype=np.float32,
    )
    valid_glt = np.all(glt_array != GLT_NODATA_VALUE, axis=-1)

    # Adjust for One based Index - make a copy to prevent decrementing multiple
    # times inside ortho_xr when applying the glt to elev
    glt_array_copy = glt_array.copy()
    glt_array_copy[valid_glt] -= 1
    out_ds[valid_glt, :] = ds_array[
        glt_array_copy[valid_glt, 1], glt_array_copy[valid_glt, 0], :
    ]
    return out_ds


def coord_vects(ds):
    """
    Calculate the Lat and Lon Vectors/Coordinate Grid.

    This function calculates the Lat and Lon Coordinate Vectors using the GLT
    and Metadata from an EMIT dataset read into xarray.

    Parameters
    ----------
    ds : xarray.Dataset
        an xarray.Dataset containing the root variable and metadata of an EMIT
        dataset.

    Returns
    -------
    lon : numpy array
        Longitude.
    lat : numpy array
        Latitude.

    """
    # Retrieve Geotransform from Metadata
    GT = ds.geotransform
    # Create Array for Lat and Lon and fill
    dim_x = ds.glt_x.shape[1]
    dim_y = ds.glt_x.shape[0]
    lon = np.zeros(dim_x)
    lat = np.zeros(dim_y)
    # Note: no rotation for EMIT Data
    for x in np.arange(dim_x):
        # Adjust coordinates to pixel-center
        x_geo = (GT[0] + 0.5 * GT[1]) + x * GT[1]
        lon[x] = x_geo
    for y in np.arange(dim_y):
        y_geo = (GT[3] + 0.5 * GT[5]) + y * GT[5]
        lat[y] = y_geo
    return lon, lat


def ortho_xr(ds, GLT_NODATA_VALUE=0, fill_value=-9999):
    """
    Use `apply_glt` to create an orthorectified xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by emit_xarray.
    GLT_NODATA_VALUE : int, optional
        No data value for the GLT tables. The default is 0.
    fill_value : int, optional
        The fill value for EMIT datasets. The default is -9999.

    Returns
    -------
    out_xr : xarray.Dataset
        an orthocorrected xarray dataset.

    """
    # Build glt_ds

    glt_ds = np.nan_to_num(
        np.stack([ds["glt_x"].data, ds["glt_y"].data], axis=-1),
        nan=GLT_NODATA_VALUE).astype(int)

    # List Variables
    var_list = list(ds.data_vars)

    # Remove flat field from data vars - the flat field is only useful with
    # additional information before orthorectification
    if "flat_field_update" in var_list:
        var_list.remove("flat_field_update")

    # Create empty dictionary for orthocorrected data vars
    data_vars = {}

    # Extract Rawspace Dataset Variable Values (Typically Reflectance)
    for var in var_list:
        raw_ds = ds[var].data
        var_dims = ds[var].dims
        # Apply GLT to dataset
        out_ds = apply_glt(raw_ds, glt_ds, GLT_NODATA_VALUE=GLT_NODATA_VALUE)

        # Mask fill values
        out_ds[out_ds == fill_value] = np.nan

        # Update variables - Only works for 2 or 3 dimensional arays
        if raw_ds.ndim == 2:
            out_ds = out_ds.squeeze()
            data_vars[var] = (["latitude", "longitude"], out_ds)
        else:
            data_vars[var] = (["latitude", "longitude", var_dims[-1]], out_ds)

        del raw_ds

    # Calculate Lat and Lon Vectors
    lon, lat = coord_vects(ds)

    # Apply GLT to elevation
    elev_ds = apply_glt(ds["elev"].data, glt_ds)
    elev_ds[elev_ds == fill_value] = np.nan

    # Delete glt_ds - no longer needed
    del glt_ds

    # Create Coordinate Dictionary
    coords = {"latitude": (["latitude"], lat),
              "longitude": (["longitude"], lon),
              **ds.coords,
              }  # unpack to add appropriate coordinates

    # Remove Unnecessary Coords
    for key in ["downtrack", "crosstrack", "lat", "lon", "glt_x", "glt_y",
                "elev"]:
        del coords[key]

    # Add Orthocorrected Elevation
    coords["elev"] = (["latitude", "longitude"], np.squeeze(elev_ds))

    # Build Output xarray Dataset and assign data_vars array attributes
    out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)

    del out_ds
    # Assign Attributes from Original Datasets
    for var in var_list:
        out_xr[var].attrs = ds[var].attrs
    out_xr.coords["latitude"].attrs = ds["lat"].attrs
    out_xr.coords["longitude"].attrs = ds["lon"].attrs
    out_xr.coords["elev"].attrs = ds["elev"].attrs

    # Add Spatial Reference in recognizable format
    out_xr.rio.write_crs(ds.spatial_ref, inplace=True)

    return out_xr


def xr_to_pygmi(xr_ds, piter=iter, showlog=print, tnames=None, metaonly=False):
    """
    Xarray to PyGMI dataset.

    Takes an EMIT dataset read into an xarray dataset using the emit_xarray
    function and convert to PyGMI dataset.

    Parameters
    ----------
    xr_ds: xarray.Dataset
        an EMIT dataset read into xarray using the emit_xarray function.

    Returns
    -------
    dat: list
        list of pygmi Data

    """
    dat = []
    var_names = list(xr_ds.data_vars)

    # Loop through variable names
    for var in var_names:
        nbands = 1
        if len(xr_ds[var].data.shape) > 2:
            nbands = xr_ds[var].data.shape[2]

        # Start building metadata
        metadata = {
            "lines": xr_ds[var].data.shape[0],
            "samples": xr_ds[var].data.shape[1],
            "bands": nbands,
            "header offset": 0,
            "file type": "ENVI Standard",
            # "data type": envi_typemap[str(xr_ds[var].data.dtype)],
            "byte order": 0,
        }

        for key in list(xr_ds.attrs.keys()):
            if key == "summary":
                metadata["description"] = xr_ds.attrs[key]
            elif key not in ["geotransform", "spatial_ref"]:
                metadata[key] = f"{{ {xr_ds.attrs[key]} }}"

        # List all variables in dataset (including coordinate variables)
        meta_vars = list(xr_ds.variables)

        # Add band parameter information to metadata (ie wavelengths/obs etc.)
        for m in meta_vars:
            if m == "wavelengths" or m == "radiance_wl":
                metadata["wavelength"] = np.array(xr_ds[m].data).astype(str).tolist()
            elif m == "fwhm" or m == "radiance_fwhm":
                metadata["fwhm"] = np.array(xr_ds[m].data).astype(str).tolist()
            elif m == "good_wavelengths":
                metadata["good_wavelengths"] = (
                    np.array(xr_ds[m].data).astype(int).tolist())
            elif m == "observation_bands":
                metadata["band names"] = np.array(xr_ds[m].data).astype(str).tolist()
            elif m == "mask_bands":
                if var == "band_mask":
                    metadata["band names"] = [
                        "packed_bands_" + bn
                        for bn in np.arange(285 / 8).astype(str).tolist()]
                else:
                    metadata["band names"] = (
                        np.array(xr_ds[m].data).astype(str).tolist()
                    )
        if "band names" not in metadata:
            if "wavelength" in metadata:
                metadata["band names"] = metadata["wavelength"]
            elif nbands == 1:
                metadata["band names"] = [var]
            else:
                metadata["band names"] = [f'{var} Band {i+1}' for i in
                                          range(nbands)]

        # Replace NaN values in each layer with fill_value
        nval = -9999
        if not metaonly:
            np.nan_to_num(xr_ds[var].data, copy=False, nan=nval)

            xrdat = xr_ds[var].data
            if len(xrdat.shape) == 2:
                xrdat = xrdat.reshape((xrdat.shape[0], xrdat.shape[1], 1))
            # rows, cols, nbands = xrdat.shape

        for bandnr in piter(range(nbands)):
            tmp = Data()
            if not metaonly:
                tmp.data = np.ma.masked_equal(xrdat[:, :, bandnr], nval)
            tmp.dataid = metadata['band names'][bandnr]
            if tnames is not None and tmp.dataid not in tnames:
                continue
            tmp.crs = CRS.from_epsg(4326)

            ymax = float(metadata['northernmost_latitude'][1:-1])
            xmin = float(metadata['westernmost_longitude'][1:-1])

            dxy = float(metadata['spatialResolution'][1:-1])

            tmp.set_transform(dxy, xmin, dxy, ymax)
            tmp.meta = metadata

            tmp.nodata = nval
            timetxt = metadata['time_coverage_start'][1:-1].strip()
            tmp.datetime = datetime.datetime.fromisoformat(timetxt)

            bmeta = tmp.metadata['Raster']
            bmeta['Sensor'] = f'EMIT {var}'

            if 'mineral_id' in var:
                bmeta['MineralNames'] = xr_ds.mineral_name.to_numpy()
                bmeta['MineralNames'] = np.insert(bmeta['MineralNames'], 0,
                                                  'None')
            if 'wavelength' in metadata:
                tmp.units = var.capitalize()
                wlen = float(metadata['wavelength'][bandnr])
                bmeta['wavelength'] = wlen

                if 'fwhm' in metadata:
                    bwidth = float(metadata['fwhm'][bandnr])
                    bmeta['WavelengthMin'] = wlen - bwidth/2
                    bmeta['WavelengthMax'] = wlen + bwidth/2
            dat.append(tmp)

    return dat


def main():
    """EMIT data."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from pygmi.misc import discrete_colorbar, getinfo

    # ifile = r"D:/EMIT/EMIT_L1B_OBS_001_20240430T101307_2412107_042.nc"
    # ifile = r"D:/EMIT/EMIT_L1B_RAD_001_20240430T101307_2412107_042.nc"
    # ifile = r"D:/EMIT/EMIT_L2A_MASK_001_20240430T101307_2412107_042.nc"
    # ifile = r"D:/EMIT/EMIT_L2A_RFL_001_20240430T101307_2412107_042.nc"
    # ifile = r"D:/EMIT/EMIT_L2A_RFLUNCERT_001_20240430T101307_2412107_042.nc"
    ifile = r"D:/EMIT/EMIT_L2B_MIN_001_20240430T101307_2412107_042.nc"
    # ifile = r"D:/EMIT/EMIT_L2B_MINUNCERT_001_20240430T101307_2412107_042.nc"
    # ifile = r"D:/EMIT/EMIT_L3_ASA_001.nc"

    getinfo()
    ds = emit_xarray(ifile)
    getinfo(1)
    # ds = emit_xarray(ifile, ortho=True)
    # getinfo(2)

    dat = xr_to_pygmi(ds)
    getinfo(3)

    for i in dat:
        fig = plt.figure(dpi=200)
        ax = fig.gca()
        plt.title(i.dataid)
        cax = ax.imshow(i.data, extent=i.extent)
        if 'mineral' in i.dataid:
            vals = np.unique(i.data)
            if np.ma.isMaskedArray(vals):
                vals = vals.compressed()
            vals = vals[~np.isnan(vals)]
            vals = vals.astype(int)

            bnds = vals.tolist() + [vals.max()+1]

            cmap = cm.viridis
            norm = colors.BoundaryNorm(bnds, cmap.N)
            cax = ax.imshow(i.data, extent=i.extent, norm=norm)

            minerals = i.metadata['Raster']['MineralNames'][vals]

            discrete_colorbar(ax, cax, i.data, minerals)
        else:
            cax = ax.imshow(i.data, extent=i.extent)
            fig.colorbar(cax)
        plt.show()


if __name__ == "__main__":
    main()

    print('Finished!')
