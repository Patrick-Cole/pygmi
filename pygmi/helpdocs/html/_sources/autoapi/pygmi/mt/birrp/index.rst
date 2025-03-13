pygmi.mt.birrp
==============

.. py:module:: pygmi.mt.birrp

.. autoapi-nested-parse::

   BIRRP -Bounded Influence Remote Reference Processing.

   BIRRP is developed by:
   Dr Alan D. Chave
   Woods Hole Oceanographic Institution
   achave@whoi.edu

   It requires an executable which must be obtained directly from Dr Chave.
   Details can be found at:
   https://www.whoi.edu/science/AOPE/people/achave/Site/Next1.html

   Conditions for the use of the BIRRP bounded influence remote reference
   magnetotelluric processing program:

      1. The robust bounded influence magnetotelluric analysis program,
         hereinafter called BIRRP, is provided on a caveat emptor basis.
         The author of BIRRP is not responsible for or culpable in the event of
         errors in processing or interpretation resulting from use of this code.
      2. No payment will be accepted by any user for data processing with BIRRP.
      3. BIRRP will not be distributed to anyone. Interested persons should be
         referred to this website.
      4. The author will be notified of any possible coding errors that are
         encountered.
      5. The author will be informed of any improvements or additions that are
         made to BIRRP.
      6. The use of BIRRP will be acknowledged in any publications and
         presentations that ensue.

   If these conditions are acceptable, send e-mail to achave@whoi.edu.
   The body of the message should state "I accept the conditions under which
   BIRRP is distributed" and copy the above six conditions.
   A gzipped tar file containing the source code will be distributed by
   return e-mail.

   Note, it will still be necessary for the end-user to compile the code.



Classes
-------

.. autoapisummary::

   pygmi.mt.birrp.BIRRP


Module Contents
---------------

.. py:class:: BIRRP(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Class to export config file for BIRRP.


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: importbirrp()

      Import a BIRRP config file.

      :rtype: None.



   .. py:method:: runbirrp()

      Save and runs a birrp config file.

      :rtype: None.



   .. py:method:: get_filename(widget)

      Get filename for a component.

      :param widget: widget whose text is set to filename..
      :type widget: widget

      :rtype: None.



   .. py:method:: nar_changed()

      Value of nar changed.

      :rtype: None.



   .. py:method:: nfil_changed()

      Value of nfil changed.

      :rtype: None.



   .. py:method:: imode_changed(indx)

      Value of imode changed.

      :param indx: Index.
      :type indx: int

      :rtype: None.



   .. py:method:: jmode_changed()

      Value of jmode changed.

      :rtype: None.



   .. py:method:: nout_changed()

      Value of nout changed.

      :rtype: None.



   .. py:method:: showrow(row, label, widget, lay)

      Show a row within a widget.

      :param row: Row number.
      :type row: int
      :param label: Row label.
      :type label: str
      :param widget: Qt widget.
      :type widget: Qt widget.
      :param lay: Form Layout.
      :type lay: QtWidgets.QFormLayout

      :rtype: None.



   .. py:method:: removerow(widget, lay)

      Remove a row.

      :param widget: Qt widget.
      :type widget: Qt widget.
      :param lay: Form Layout.
      :type lay: QtWidgets.QFormLayout

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



