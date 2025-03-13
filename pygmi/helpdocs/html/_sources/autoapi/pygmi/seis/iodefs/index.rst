pygmi.seis.iodefs
=================

.. py:module:: pygmi.seis.iodefs

.. autoapi-nested-parse::

   Import and export seismology data.



Classes
-------

.. autoapisummary::

   pygmi.seis.iodefs.ImportSeisan
   pygmi.seis.iodefs.ImportGenericFPS
   pygmi.seis.iodefs.ExportSeisan
   pygmi.seis.iodefs.ExportCSV
   pygmi.seis.iodefs.ExportSummary
   pygmi.seis.iodefs.FilterSeisan


Functions
---------

.. autoapisummary::

   pygmi.seis.iodefs.sform
   pygmi.seis.iodefs.str2float
   pygmi.seis.iodefs.str2int
   pygmi.seis.iodefs.importmacro
   pygmi.seis.iodefs.importnordic
   pygmi.seis.iodefs.importseiscomp
   pygmi.seis.iodefs.importxlsx
   pygmi.seis.iodefs.read_record_type_1
   pygmi.seis.iodefs.read_record_type_2
   pygmi.seis.iodefs.read_record_type_3
   pygmi.seis.iodefs.read_record_type_4
   pygmi.seis.iodefs.read_record_type_phase
   pygmi.seis.iodefs.read_record_type_5
   pygmi.seis.iodefs.read_record_type_6
   pygmi.seis.iodefs.read_record_type_e
   pygmi.seis.iodefs.read_record_type_f
   pygmi.seis.iodefs.read_record_type_h
   pygmi.seis.iodefs.read_record_type_i
   pygmi.seis.iodefs.read_record_type_m
   pygmi.seis.iodefs.merge_m
   pygmi.seis.iodefs.read_record_type_p
   pygmi.seis.iodefs.mercalli
   pygmi.seis.iodefs.xlstomacro


Module Contents
---------------

.. py:function:: sform(strform, val, tmp, col1, col2=None, nval=-999)

   Format strings.

   Formats strings according with a mod for values containing the value -999
   or None. In that case it will output spaces instead. In the case of strings
   being output, they are truncated to fit the format statement. This routine
   also puts the new strings in the correct columns

   :param strform: This string must be of the form {0:4.1f}, where 4.1f can be changed.
   :type strform: python format string
   :param val: input value
   :type val: float, int, str
   :param tmp: Input string
   :type tmp: str
   :param col1: start column (1 is first column)
   :type col1: int
   :param col2: end column. The default is None.
   :type col2: int
   :param nval: null value which gets substituted by spaces. The default is -999.
   :type nval: float, int

   :returns: **tmp** -- Output formatted string.
   :rtype: str


.. py:function:: str2float(inp)

   Convert a number  float, or returns NaN.

   :param inp: string with a float in it
   :type inp: str

   :returns: **output** -- float or np.nan
   :rtype: float


.. py:function:: str2int(inp)

   Convert a number to integer, or returns NaN.

   :param inp: string with an integer in it
   :type inp: str

   :returns: **output** -- integer or np.nan
   :rtype: int


.. py:class:: ImportSeisan(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to import SEISAN and SeisComP data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:function:: importmacro(ifile)

   Import macro format.

   1.  Line
       Location, GMT time, Local time. Format a30,i4,1x,2i2,1x,2i2,1x,i2,
       'GMT',1x,i4,1x,2i2,1x,2i2,1x,i2,1x,'Local time'
   2.  Line Comments
   3.  Line Observations: Latitude, Longitude,intensity, code for scale,
       postal code or similar, location,Format 2f10.4,f5.1,1x,a3,1x,a10,2x,a.
       Note the postal code is an ascii string and left justified (a10).

   :param ifile: Input macro file.
   :type ifile: str

   :returns: **gdf1** -- List of locations with intensities.
   :rtype: GeoPandas dataframe


.. py:function:: importnordic(ifile, showlog=print)

   Import Nordic and Nordic2 data.

   :param ifile: Input file to import.
   :type ifile: str
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: **dat** -- SEISAN Data.
   :rtype: list


.. py:function:: importseiscomp(ifile, showlog=print, prefmag='MLv')

   Import SeisComp data.

   :param ifile: Input file to import.
   :type ifile: str
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: **sdat** -- SEISAN Data.
   :rtype: list


.. py:function:: importxlsx(ifile, showlog=print)

   Import Excel summary.

   :param ifile: Input file to import.
   :type ifile: str
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: **sdat** -- SEISAN Data.
   :rtype: list


.. py:function:: read_record_type_1(i)

   Read record type 1.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN 1 record.
   :rtype: sdt.seisan_1


.. py:function:: read_record_type_2(i)

   Read record type 2.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN 2 record.
   :rtype: sdt.seisan_2


.. py:function:: read_record_type_3(i)

   Read record type 3.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN 4 record.
   :rtype: sdt.seisan_4


.. py:function:: read_record_type_4(i)

   Read record type 4.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN 4 record.
   :rtype: sdt.seisan_4


.. py:function:: read_record_type_phase(i)

   Read record type phase (nordic2 type 4).

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN 4 record.
   :rtype: sdt.seisan_4


.. py:function:: read_record_type_5(i)

   Read record type 5.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN 5 record.
   :rtype: sdt.seisan_5


.. py:function:: read_record_type_6(i)

   Read record type 6.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN 6 record.
   :rtype: sdt.seisan_6


.. py:function:: read_record_type_e(i)

   Read record type E.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN E record.
   :rtype: sdt.seisan_E


.. py:function:: read_record_type_f(i)

   Read record type F.

   :param i: String to read from.
   :type i: str

   :returns: **out** -- Dictionary with a SEISAN F record.
   :rtype: dictionary


.. py:function:: read_record_type_h(i)

   Read record type H.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN H record.
   :rtype: sdt.seisan_H


.. py:function:: read_record_type_i(i)

   Read record type I.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN I record.
   :rtype: sdt.seisan_I


.. py:function:: read_record_type_m(i)

   Read record type M.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN M record.
   :rtype: sdt.seisan_M


.. py:function:: merge_m(rec1, rec2)

   Merge M records.

   :param rec1: SEISAN M record.
   :type rec1: sdt.seisan_M
   :param rec2: SEISAN M record.
   :type rec2: sdt.seisan_M

   :returns: **rec1** -- SEISAN M record.
   :rtype: sdt.seisan_M


.. py:function:: read_record_type_p(i)

   Read record type P.

   :param i: String to read from.
   :type i: str

   :returns: **tmp** -- SEISAN P record.
   :rtype: sdt.seisan_P


.. py:class:: ImportGenericFPS(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to import Generic Fault Plane Solution data.

   This is stored in a csv file.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



.. py:class:: ExportSeisan(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to export SEISAN data.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: run(filename=None)

      Entry point into the routine, used to run context menu item.

      :rtype: None.



   .. py:method:: write_record_type_1(data)

      Write record type 1.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_2(data)

      Write record type 2.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_3(data)

      Write record type 3.

      This changes depending on the preceding line.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_4(data)

      Write record type 4.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_phase(data)

      Write record type 4.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_5(data)

      Write record type 5.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_6(data)

      Write record type 6.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_7()

      Write record type 7.

      :rtype: None.



   .. py:method:: write_record_type_e(data)

      Write record type E.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_f(data)

      Write record type F.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_h(data)

      Write record type H.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_i(data)

      Write record type I.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_m(data)

      Write record type M.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



   .. py:method:: write_record_type_p(data)

      Write record type P.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :rtype: None.



.. py:class:: ExportCSV(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI export seismic data to CSV.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



   .. py:method:: write_record_type_1(data)

      Write record type 1.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_2(data)

      Write record type 2.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_3(tmp)

      Write record type 3.

      This changes depending on the preceding line.


      :param tmp: Data string.
      :type tmp: str

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_4(data)

      Write record type 4.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmpfin** -- List of output string.
      :rtype: list



   .. py:method:: write_record_type_5(data)

      Write record type 5.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_6(data)

      Write record type 6.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_7()

      Write record type 7.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_e(data)

      Write record type E.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_f(data)

      Write record type F.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_h(data)

      Write record type H.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_i(data)

      Write record type I.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_m(data)

      Write record type M.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



   .. py:method:: write_record_type_p(data)

      Write record type P.

      :param data: Dictionary of record types.
      :type data: Dictionary

      :returns: **tmp** -- Output string.
      :rtype: str



.. py:class:: ExportSummary(parent=None)

   Bases: :py:obj:`pygmi.misc.ContextModule`


   GUI to export a seismic data summary.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: run()

      Entry point into the routine, used to run context menu item.

      :rtype: None.



.. py:function:: mercalli(mag)

   Return Mercalli index.

   :param mag: Local magnitude.
   :type mag: float

   :returns: **merc** -- Mercalli index
   :rtype: str


.. py:class:: FilterSeisan(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to filter seismic data events.

   This filters data using thresholds.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: dind_click(state)

      Check checkboxes.

      :param state: State of checkbox.
      :type state: int

      :rtype: None.



   .. py:method:: rectype_init(txt)

      Change combo.

      :param txt: Text.
      :type txt: str

      :rtype: None.



   .. py:method:: recdesc_init(txt)

      Change Description.

      :param txt: Text.
      :type txt: str

      :rtype: None.



   .. py:method:: get_limits()

      Get limits for SEISAN data.

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



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



.. py:function:: xlstomacro()

   Convert an excel file to macro file.

   1.  Line
       Location, GMT time, Local time. Format a30,i4,1x,2i2,1x,2i2,1x,i2,
       'GMT',1x,i4,1x,2i2,1x,2i2,1x,i2,1x,'Local time'
   2.  Line Comments
   3.  Line Observations: Latitude, Longitude,intensity, code for scale,
       postal code or similar, location,Format 2f10.4,f5.1,1x,a3,1x,a10,2x,a.
       Note the postal code is an ascii string and left justified (a10).



