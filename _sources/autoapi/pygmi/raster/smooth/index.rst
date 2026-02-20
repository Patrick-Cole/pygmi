pygmi.raster.smooth
===================

.. py:module:: pygmi.raster.smooth

.. autoapi-nested-parse::

   Routines to smooth raster data.



Classes
-------

.. autoapisummary::

   pygmi.raster.smooth.Smooth


Functions
---------

.. autoapisummary::

   pygmi.raster.smooth.mov_win_filt
   pygmi.raster.smooth.filters2d


Module Contents
---------------

.. py:class:: Smooth(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Smooth rasters.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


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



   .. py:method:: choosefilter()

      Section to choose the filter.

      :rtype: None.



   .. py:method:: updatetable()

      Update table.

      :rtype: None.



   .. py:method:: msgbox(title, message)

      Message box.

      :param title: Title for message box.
      :type title: str
      :param message: Text for message box.
      :type message: str

      :rtype: None.



.. py:function:: mov_win_filt(dat, fmat, itype, box_x=5, box_y=5, rad=5, sigma=5, showlog=print, piter=iter)

   Apply moving window filter function to data.

   :param dat: Data for a PyGMI raster dataset.
   :type dat: numpy masked array.
   :param fmat: Filter matrix type.
   :type fmat: str
   :param itype: Filter type. Can be '2D Mean' or '2D Median'.
   :type itype: str
   :param box_x: number of columns for box, by default 5
   :type box_x: int, optional
   :param box_y: number of rows for box, by default 5
   :type box_y: int, optional
   :param rad: Radius of disc window, by default 5
   :type rad: int, optional
   :param sigma: Standard deviation, by default 5
   :type sigma: int, optional
   :param showlog: Routine to show text messages. The default is print.
   :type showlog: function, optional
   :param piter: progress bar iterable, default is iter.
   :type piter: function, optional

   :returns: **out** -- Data for a PyGMI raster dataset.
   :rtype: numpy masked array


.. py:function:: filters2d(filtertype, sze, *sigma)

   Filter 2D.

   These filter definitions have been translated from the octave function
   'fspecial'.

   Copyright (C) 2005 Peter Kovesi

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
   02110-1301, USA.

   FSPECIAL - Create spatial filters for image processing

   Usage:  f = fspecial(filtertype, optional parameters)

   filtertype can be

   |   'average'   - Rectangular averaging filter
   |   'disc'      - Circular averaging filter.
   |   'gaussian'  - Gaussian filter.

   The parameters that need to be specified depend on the filtertype

   Examples of use and associated default values:

   |   f = fspecial('average',sze)           sze can be a 1 or 2 vector
   |                                         default is [3 3].
   |   f = fspecial('disk',radius)           default radius = 5
   |   f = fspecial('gaussian',sze, sigma)   default sigma is 0.5

   Where sze is specified as a single value the filter will be square.

   Author:   Peter Kovesi <pk@csse.uwa.edu.au>
   Keywords: image processing, spatial filters
   Created:  August 2005

   :param filtertype: Type of filter. Can be 'average', 'disc' or 'gaussian'.
   :type filtertype: str
   :param sze: This is a integer radius for 'disc' or a vector containing rows and
               columns otherwise.
   :type sze: numpy array or integer)
   :param sigma: numpy array containing std deviation. Used in 'gaussian'.
   :type sigma: numpy array

   :returns: **f** -- Returns the filter to be used.
   :rtype: numpy array


