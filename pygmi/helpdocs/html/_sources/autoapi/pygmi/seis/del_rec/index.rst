pygmi.seis.del_rec
==================

.. py:module:: pygmi.seis.del_rec

.. autoapi-nested-parse::

   Delete SEISAN records.



Classes
-------

.. autoapisummary::

   pygmi.seis.del_rec.DeleteRecord
   pygmi.seis.del_rec.Quarry


Functions
---------

.. autoapisummary::

   pygmi.seis.del_rec.import_for_plots


Module Contents
---------------

.. py:class:: DeleteRecord(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to delete records.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: delrec(ifile)

      Delete record.

      :param ifile: Input filename.
      :type ifile: str

      :rtype: None.



.. py:class:: Quarry(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to implement quarry event filtering.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: calcrq2()

      Calculate the Rq value.

      :returns: **newevents** -- New events
      :rtype: list



   .. py:method:: calcrq2b()

      Calculate the Rq value.

      :returns: **newevents** -- New events
      :rtype: list



   .. py:method:: randrq(nmax, nstep, nrange, day)

      Calculate random Rq values.

      :param nmax: DESCRIPTION.
      :type nmax: int
      :param nstep: DESCRIPTION.
      :type nstep: int
      :param nrange: DESCRIPTION.
      :type nrange: list
      :param day: DESCRIPTION.
      :type day: tuple

      :returns: **rperc** -- Percentiles
      :rtype: list



   .. py:method:: randrqb(N1, day, num)

      Calculate random Rq values.

      :param N1: DESCRIPTION.
      :type N1: TYPE
      :param day: DESCRIPTION.
      :type day: tuple
      :param num: DESCRIPTION.
      :type num: int

      :returns: **rperc** -- Percentiles
      :rtype: list



.. py:function:: import_for_plots(ifile, dind='R')

   Import data to plot.

   :param ifile: Input file name.
   :type ifile: str
   :param dind: Distance indicator. The default is 'R'.
   :type dind: str, optional

   :returns: **datd** -- Output data.
   :rtype: dictionary


