pygmi.grav.dataprep
===================

.. py:module:: pygmi.grav.dataprep

.. autoapi-nested-parse::

   A set of data processing routines for gravity.



Classes
-------

.. autoapisummary::

   pygmi.grav.dataprep.MyMplCanvas
   pygmi.grav.dataprep.PlotDrift
   pygmi.grav.dataprep.ProcessData


Functions
---------

.. autoapisummary::

   pygmi.grav.dataprep.gravcor
   pygmi.grav.dataprep.geocentric_radius
   pygmi.grav.dataprep.theoretical_gravity
   pygmi.grav.dataprep.atmospheric_correction
   pygmi.grav.dataprep.height_correction
   pygmi.grav.dataprep.spherical_bouguer
   pygmi.grav.dataprep.time_convert


Module Contents
---------------

.. py:class:: MyMplCanvas(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: update_raster(drift)

      Update the raster plot.

      :param drift: Dictionary containing information for drift plots.
      :type drift: dict

      :rtype: None.



.. py:class:: PlotDrift(parent=None, data=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Plot Raster Class.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


.. py:class:: ProcessData(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Process Gravity Data.

   This class processes gravity data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: acceptall(nodialog)

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



   .. py:method:: calcbase()

      Calculate local base station value.

      Ties in the local base station to a known absolute base station.

      :rtype: None.



.. py:function:: gravcor(pdat, basethres, kstat='None', absbase=978032.67715, dens=2670, showlog=print)

   Gravity corrections.

   :param pdat: Gravity data.
   :type pdat: Pandas DataFrame
   :param basethres: Threshold for base station numbers.
   :type basethres: float
   :param kstat: Known base station number.
   :type kstat: float
   :param absbase: Known base station absolute gravity.
   :type absbase: float
   :param dens: Background Density (kg/m3).
   :type dens: float
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: * **pdat** (*Pandas DataFrame*) -- Gravity data.
             * **drift** (*dict*) -- Dictionary containing information for drift plots.


.. py:function:: geocentric_radius(lat)

   Geocentric radius calculation.

   Calculate the distance from the Earth's center to a point on the spheroid
   surface at a specified geodetic latitude.

   :param lat: Latitude in radians
   :type lat: numpy array

   :returns: **R** -- Array of radii.
   :rtype: Numpy array


.. py:function:: theoretical_gravity(lat)

   Calculate the theoretical gravity.

   :param lat: Latitude in radians
   :type lat: numpy array

   :returns: **gT** -- Array of theoretical gravity values.
   :rtype: numpy array


.. py:function:: atmospheric_correction(h)

   Calculate the atmospheric correction.

   :param h: Heights relative to ellipsoid (GPS heights).
   :type h: numpy array

   :returns: **gATM** -- Atmospheric correction
   :rtype: numpy array.


.. py:function:: height_correction(lat, h)

   Calculate height correction.

   :param lat: Latitude in radians.
   :type lat: numpy array
   :param h: Heights relative to ellipsoid (GPS heights).
   :type h: numpy array

   :returns: **gHC** -- Height corrections
   :rtype: numpy array


.. py:function:: spherical_bouguer(h, dens)

   Calculate spherical Bouguer.

   :param h: Heights relative to ellipsoid (GPS heights).
   :type h: numpy array
   :param dens: Density.
   :type dens: float

   :returns: **gSB** -- Spherical Bouguer correction.
   :rtype: numpy array


.. py:function:: time_convert(x)

   Convert hh:mm:ss to seconds.

   :param x: Time in hh:mm:ss.
   :type x: str

   :returns: Time in seconds.
   :rtype: float


