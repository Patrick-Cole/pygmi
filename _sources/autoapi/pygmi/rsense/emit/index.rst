pygmi.rsense.emit
=================

.. py:module:: pygmi.rsense.emit

.. autoapi-nested-parse::

   EMIT is used to import EMIT satellite data into PyGMI.

   It uses code by Erik Bolch, ebolch@contractor.usgs.gov



Functions
---------

.. autoapisummary::

   pygmi.rsense.emit.emit_xarray
   pygmi.rsense.emit.apply_glt
   pygmi.rsense.emit.coord_vects
   pygmi.rsense.emit.ortho_xr
   pygmi.rsense.emit.xr_to_pygmi
   pygmi.rsense.emit.main


Module Contents
---------------

.. py:function:: emit_xarray(filepath, ortho=False, qmask=None, unpackedbmask=None)

   EMIT xarray.

   This function utilizes other functions in this module to streamline
   opening an EMIT dataset as an xarray.Dataset.

   :param filepath: a file path to an EMIT netCDF file.
   :type filepath: str
   :param ortho: Whether to orthorectify the dataset or leave in crosstrack/downtrack
                 coordinates. The default is False.
   :type ortho: bool, optional
   :param qmask: Output from the quality_mask function used to mask
                 pixels based on quality flags selected in that function. Any
                 non-orthorectified array with the proper crosstrack and downtrack
                 dimensions can also be used. The default is None.
   :type qmask: numpy array, optional
   :param unpackedbmask: From the band_mask function, used to mask band-specific pixels that
                         have been interpolated. The default is None.
   :type unpackedbmask: numpy array, optional

   :returns: **out_xr** -- Dataset constructed based on the parameters provided.
   :rtype: xarray.Dataset


.. py:function:: apply_glt(ds_array, glt_array, fill_value=-9999, GLT_NODATA_VALUE=0)

   Apply GLT.

   This function applies the GLT array to a numpy array of either 2 or 3
   dimensions.

   :param ds_array: A numpy array of the desired variable.
   :type ds_array: numpy array
   :param glt_array: A GLT array constructed from EMIT GLT data.
   :type glt_array: GLT array
   :param fill_value: Fill value. The default is -9999.
   :type fill_value: int, optional
   :param GLT_NODATA_VALUE: GLT nodata value. The default is 0.
   :type GLT_NODATA_VALUE: int, optional

   :returns: **out_ds** -- a numpy array of orthorectified data.
   :rtype: numpy array


.. py:function:: coord_vects(ds)

   Calculate the Lat and Lon Vectors/Coordinate Grid.

   This function calculates the Lat and Lon Coordinate Vectors using the GLT
   and Metadata from an EMIT dataset read into xarray.

   :param ds: an xarray.Dataset containing the root variable and metadata of an EMIT
              dataset.
   :type ds: xarray.Dataset

   :returns: * **lon** (*numpy array*) -- Longitude.
             * **lat** (*numpy array*) -- Latitude.


.. py:function:: ortho_xr(ds, GLT_NODATA_VALUE=0, fill_value=-9999)

   Use `apply_glt` to create an orthorectified xarray dataset.

   :param ds: Dataset produced by emit_xarray.
   :type ds: xarray.Dataset
   :param GLT_NODATA_VALUE: No data value for the GLT tables. The default is 0.
   :type GLT_NODATA_VALUE: int, optional
   :param fill_value: The fill value for EMIT datasets. The default is -9999.
   :type fill_value: int, optional

   :returns: **out_xr** -- an orthocorrected xarray dataset.
   :rtype: xarray.Dataset


.. py:function:: xr_to_pygmi(xrds, piter=iter, showlog=print, tnames=None, metaonly=False)

   Xarray to PyGMI dataset.

   Takes an EMIT dataset read into an xarray dataset using the emit_xarray
   function and convert to PyGMI dataset.

   :param xrds: an EMIT dataset read into xarray using the emit_xarray function.
   :type xrds: xarray.Dataset
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param tnames: list of band names to import, in order. The default is None.
   :type tnames: list, optional
   :param metaonly: Retrieve only the metadata for the file. The default is False.
   :type metaonly: bool, optional

   :returns: **dat** -- list of pygmi.raster.datatypes.Data
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: main()

   EMIT data.


