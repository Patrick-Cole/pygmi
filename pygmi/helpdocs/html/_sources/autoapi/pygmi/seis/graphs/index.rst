pygmi.seis.graphs
=================

.. py:module:: pygmi.seis.graphs

.. autoapi-nested-parse::

   Plot Seismology Data.

   This module provides a variety of methods to plot raster data via the context
   menu.



Classes
-------

.. autoapisummary::

   pygmi.seis.graphs.MyMplCanvas
   pygmi.seis.graphs.GraphWindow
   pygmi.seis.graphs.PlotQC
   pygmi.seis.graphs.PlotIso
   pygmi.seis.graphs.PlotTempB


Functions
---------

.. autoapisummary::

   pygmi.seis.graphs.contourtopoly
   pygmi.seis.graphs.import_for_plots
   pygmi.seis.graphs.eigsorted
   pygmi.seis.graphs.bvalue
   pygmi.seis.graphs.fmd
   pygmi.seis.graphs.maxc


Module Contents
---------------

.. py:class:: MyMplCanvas(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Matplotlib canvas widget for the actual plot.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: update_ellipse(datd, dats, nodepth=False)

      Update error ellipse plot.

      :param datd: Dictionary containing latitudes and longitudes
      :type datd: dictionary
      :param dats: Data list.
      :type dats: list
      :param nodepth: Flag to determine if there are depths. The default is False.
      :type nodepth: bool, optional

      :rtype: None.



   .. py:method:: update_hexbin(data1, data2, *, xlbl='Time', ylbl='ML', xbin=None, xrng=None)

      Update the hexbin plot.

      :param data1: raster dataset to be used
      :type data1: numpy array
      :param data2: raster dataset to be used
      :type data2: numpy array
      :param xlbl: X-axis label. The default is 'Time'.
      :type xlbl: str, optional
      :param ylbl: Y-axis label. The default is 'ML'.
      :type ylbl: str, optional
      :param xbin: Number of bins in the x direction. The default is None.
      :type xbin: int, optional
      :param xrng: X-range. The default is None.
      :type xrng: list, optional

      :rtype: None.



   .. py:method:: update_hist(data1, *, xlbl='Data Value', ylbl='Number of Observations', bins='doane', rng=None)

      Update the histogram plot.

      :param data1: raster dataset to be used
      :type data1: numpy array.
      :param xlbl: X-axis label. The default is 'Data Value'.
      :type xlbl: str, optional
      :param ylbl: Y-axis label. The default is 'Number of Observations'.
      :type ylbl: str, optional
      :param bins: Number of bins or binning strategy. See matplotlib.pyplot.hist.
                   The default is 'doane'.
      :type bins: int or str, optional
      :param rng: Bin range. The default is None.
      :type rng: tuple or None, optional

      :rtype: None.



   .. py:method:: update_bvalue(data1a, bins='doane')

      Update the b value plot.

      :param data1a: Data array.
      :type data1a: numpy array
      :param bins: Number of bins or binning strategy. See matplotlib.pyplot.hist.
                   The default is 'doane'.
      :type bins: int or str, optional

      :rtype: None.



   .. py:method:: update_pres(data1, phase='P')

      Update the plot.

      :param data1: Data array.
      :type data1: numpy array
      :param phase: Phase. The default is 'P'.
      :type phase: str, optional

      :rtype: None.



   .. py:method:: update_residual(dat, res='ML')

      Update the residual plot.

      :param data1: Data array.
      :type data1: numpy array
      :param res: Response type. The default is 'ML'.
      :type res: str, optional

      :rtype: None.



   .. py:method:: update_wadati(dat, min_wad=5, min_vps=1.53, max_vps=1.93)

      Update the wadati plot.

      :param dat: List of events.
      :type dat: list
      :param min_wad: Minimum data length for plot. The default is 5.
      :type min_wad: int, optional
      :param min_vps: Minimum VPS. The default is 1.53.
      :type min_vps: float, optional
      :param max_vps: Maximum VPS. The default is 1.93.
      :type max_vps: float, optional

      :rtype: None.



   .. py:method:: update_isohull(datd)

      Update isoseismic plot using convex hull method.

      :param datd: Macroseismic data.
      :type datd: GeoDatFrame

      :rtype: None.



   .. py:method:: update_isocontour(datd)

      Update isoseismic plot using contours.

      :param datd: Macroseismic data.
      :type datd: GeoDatFrame

      :rtype: None.



   .. py:method:: update_tempb(btot, datetot)

      Update temporal b value plot.

      :param btot: Array of b values.
      :type btot: numpy array
      :param datetot: list of dates.
      :type datetot: list

      :rtype: None.



.. py:class:: GraphWindow(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   Graph Window class.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: save_shp()

      Save shapefile.

      :rtype: None.



.. py:class:: PlotQC(parent=None)

   Bases: :py:obj:`GraphWindow`


   GUI to plot QC graphs.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



   .. py:method:: save_shp()

      Save shapefile.

      :returns: True if successful, False otherwise.
      :rtype: bool



.. py:class:: PlotIso(parent=None)

   Bases: :py:obj:`GraphWindow`


   GUI to plot isolines.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_band()

      Combo box to choose band.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



   .. py:method:: save_shp()

      Save shapefile.

      :returns: True if successful, False otherwise.
      :rtype: bool



.. py:class:: PlotTempB(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to plot temporal b-values.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: change_window()

      Edit box to change window length.

      :rtype: None.



   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



   .. py:method:: save_shp()

      Save shapefile.

      :returns: True if successful, False otherwise.
      :rtype: bool



.. py:function:: contourtopoly(cntr)

   Convert Matplotlib contours to Polygons.

   :param cntr: Contour collection.
   :type cntr: Matplotlib countour

   :returns: **plist** -- List of Polygon objects.
   :rtype: list


.. py:function:: import_for_plots(dat)

   Import data to plot.

   :param dat: List of events.
   :type dat: list

   :returns: **datd** -- Dictionary of data to plot.
   :rtype: dictionary


.. py:function:: eigsorted(cov)

   Calculate and sort eigenvalues.

   :param cov: matrix to perform calculations on.
   :type cov: numpy array

   :returns: * **vals** (*numpy array*) -- Sorted eigenvalues.
             * **vecs** (*numpy array*) -- Sorted eigenvectors.


.. py:function:: bvalue(data1a, mbin=0.1, bins='doane')

   Update the b value plot.

   :param data1a: Data array.
   :type data1a: numpy array
   :param bins: Number of bins or binning strategy. See matplotlib.pyplot.hist.
                The default is 'doane'.
   :type bins: int or str, optional

   :returns: **out** -- Dictionary containing 'a-value', 'b-value' etc.
   :rtype: dict


.. py:function:: fmd(mag, mbin=0.1)

   Frequency magnitude distribution.

   Mignan, A. & Woessner, Jochen. (2012). Estimating the magnitude of
   completeness for earthquake catalogs. Community Online Resource for
   Statistical Seismicity Analysis. 10.5078/corssa-00180805.

   :param mag: Data array of magnitudes.
   :type mag: numpy array
   :param mbin: Length of magnitude bin in FMD. The default is 0.1.
   :type mbin: float, optional

   :returns: **res** -- Dictionary containing M  vs cumulative and non cumulative counts.
   :rtype: dict


.. py:function:: maxc(mag, mbin=0.1)

   MAXC method to find magnitude of completeness.

   Mignan, A. & Woessner, Jochen. (2012). Estimating the magnitude of
   completeness for earthquake catalogs. Community Online Resource for
   Statistical Seismicity Analysis. 10.5078/corssa-00180805.

   :param mag: Data array of magnitudes.
   :type mag: numpy array
   :param mbin: Length of magnitude bin in FMD. The default is 0.1.
   :type mbin: float, optional

   :returns: **Mc** -- Magnitude of completeness.
   :rtype: float


