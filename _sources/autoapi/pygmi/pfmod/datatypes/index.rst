pygmi.pfmod.datatypes
=====================

.. py:module:: pygmi.pfmod.datatypes

.. autoapi-nested-parse::

   Class for data types.



Classes
-------

.. autoapisummary::

   pygmi.pfmod.datatypes.LithModel


Module Contents
---------------

.. py:class:: LithModel

   Lithological Model Data.

   This is the main data structure for the modelling program

   .. attribute:: mlut

      colour table for lithologies

      :type: dictionary

   .. attribute:: numx

      number of columns per layer in model

      :type: int

   .. attribute:: numy

      number of rows per layer in model

      :type: int):

   .. attribute:: numz

      number of layers in model

      :type: int

   .. attribute:: dxy

      dimension of cubes in the x and y directions

      :type: float

   .. attribute:: d_z

      dimension of cubes in the z direction

      :type: float

   .. attribute:: lith_index

      3D array of lithological indices.

      :type: numpy array

   .. attribute:: xrange

      minimum and maximum x coordinates

      :type: list

   .. attribute:: yrange

      minimum and maximum y coordinates

      :type: list

   .. attribute:: zrange

      minimum and maximum z coordinates

      :type: list

   .. attribute:: griddata

      dictionary of Data classes with raster data

      :type: dictionary

   .. attribute:: custprofx

      custom profile x coordinates

      :type: dictionary

   .. attribute:: custprofy

      custom profile y coordinates

      :type: dictionary

   .. attribute:: profpics

      profile pictures

      :type: dictionary

   .. attribute:: lith_list

      list of lithologies

      :type: dictionary

   .. attribute:: lith_list_reverse

      reverse lookup for lith_list

      :type: dictionary

   .. attribute:: mht

      height of magnetic sensor

      :type: float

   .. attribute:: ght

      height of gravity sensor

      :type: float

   .. attribute:: gregional

      gravity regional correction

      :type: float


   .. py:method:: lithold_to_lith(nodtm=False, pbar=None)

      Transfers an old lithology to the new one, using update parameters.

      :param nodtm: Flag for a DTM. The default is False.
      :type nodtm: bool, optional
      :param pbar: Progressbar. The default is None.
      :type pbar: pygmi.misc.ProgressBar, optional

      :rtype: None.



   .. py:method:: dtm_to_lith(pbar=None)

      Assign the DTM to the model.

      This means creating nodata values in areas above the DTM. These values
      are assigned a lithology of -1.

      :param pbar: Progressbar. The default is None.
      :type pbar: pygmi.misc.ProgressBar, optional

      :rtype: None.



   .. py:method:: init_grid(data)

      Initialize raster variables in the Data class.

      :param data: Masked array containing raster data.
      :type data: numpy array

      :returns: **grid** -- PyGMI raster dataset.
      :rtype: pygmi.raster.datatypes.Data



   .. py:method:: init_calc_grids()

      Initialize mag and gravity from the model.

      :rtype: None.



   .. py:method:: is_modified(modified=True)

      Update modified flag.

      :param modified: Flag for whether the lithology has been modified. The default is
                       True.
      :type modified: bool, optional

      :rtype: None.



   .. py:method:: update(cols, rows, layers, utlx, utly, utlz, dxy, d_z, mht=-1, ght=-1, usedtm=True, pbar=None)

      Update the local variables for the LithModel class.

      :param cols: Number of columns per layer in model.
      :type cols: int
      :param rows: Number of rows per layer in model.
      :type rows: int
      :param layers: Number of layers in model.
      :type layers: int
      :param utlx: Upper top left (NW) x coordinate.
      :type utlx: float
      :param utly: Upper top left (NW) y coordinate.
      :type utly: float
      :param utlz: Upper top left (NW) z coordinate.
      :type utlz: float
      :param dxy: Dimension of cubes in the x and y directions.
      :type dxy: float
      :param d_z: Dimension of cubes in the z direction.
      :type d_z: float
      :param mht: Height of magnetic sensor. The default is -1.
      :type mht: float, optional
      :param ght: Height of gravity sensor. The default is -1.
      :type ght: float, optional
      :param usedtm: Flag to use a DTM. The default is True.
      :type usedtm: bool, optional
      :param pbar: Progressbar. The default is None.
      :type pbar: pygmi.misc.ProgressBar, optional

      :rtype: None.



   .. py:method:: update_lithlist()

      Update lith_list from local variables.

      :rtype: None.



   .. py:method:: update_lith_list_reverse()

      Update the lith_list reverse lookup.

      It must be run at least once before using lith_list_reverse.

      :rtype: None.



