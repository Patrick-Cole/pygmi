pygmi.raster.reproj
===================

.. py:module:: pygmi.raster.reproj

.. autoapi-nested-parse::

   Reprojection functions.



Classes
-------

.. autoapisummary::

   pygmi.raster.reproj.GroupProj


Functions
---------

.. autoapisummary::

   pygmi.raster.reproj.data_reproject
   pygmi.raster.reproj.getepsgcodes


Module Contents
---------------

.. py:class:: GroupProj(title='Projection', parent=None)

   Bases: :py:obj:`PyQt5.QtWidgets.QWidget`


   Group Projection GUI widget.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional
   :param title: Title for QGroupBox - self.gbox.
   :type title: str


   .. py:method:: set_current(wkt)

      Set new WKT for current option.

      :param wkt: Well Known Text descriptions for coordinates (WKT).
      :type wkt: str

      :rtype: None.



   .. py:method:: combo_datum_change()

      Change datum combo.

      :rtype: None.



   .. py:method:: combo_change()

      Change Combo.

      :rtype: None.



.. py:function:: data_reproject(data, ocrs, otransform=None, orows=None, ocolumns=None, icrs=None, showlog=print, forcereproj=False)

   Reproject dataset.

   :param data: PyGMI dataset.
   :type data: pygmi.raster.datatypes.Data
   :param ocrs: output crs.
   :type ocrs: CRS
   :param otransform: Output affine transform. The default is None.
   :type otransform: Affine, optional
   :param orows: output rows. The default is None.
   :type orows: int, optional
   :param ocolumns: output columns. The default is None.
   :type ocolumns: int, optional
   :param icrs: input crs. The default is None.
   :type icrs: CRS, optional
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param forcereproj: Force a reprojection, the default is False.
   :type forcereproj: bool, optional

   :returns: **data2** -- Reprojected dataset.
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: getepsgcodes()

   Routine used to get a list of EPSG codes.

   :returns: **pcodes** -- Dictionary of codes per projection in WKT format.
   :rtype: dictionary


