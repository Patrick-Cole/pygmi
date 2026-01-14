pygmi.main
==========

.. py:module:: pygmi.main

.. autoapi-nested-parse::

   Main module for PyGMI.

   This module comprises a series of classes which are responsible for the primary
   interface to the software. Credit must go to PyQt's examples, upon which some
   of this was originally based.

   Although the main interface is controlled here, the content of the menus and
   routines is not. The menus and corresponding classes are found within the
   pygmi packages.



Classes
-------

.. autoapisummary::

   pygmi.main.Arrow
   pygmi.main.DiagramItem
   pygmi.main.DiagramScene
   pygmi.main.MainWidget
   pygmi.main.Startup


Functions
---------

.. autoapisummary::

   pygmi.main.main


Module Contents
---------------

.. py:class:: Arrow(start_item, end_item, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsLineItem`


   Class responsible for drawing arrows on the main interface.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget optional
   :param start_item: Starting DiagramItem object.
   :type start_item: DiagramItem
   :param end_item: End DiagramItem object.
   :type end_item: DiagramItem

   .. attribute:: arrow_head

      Arrow head polygon.

      :type: QPolygonF

   .. attribute:: my_start_item

      Starting DiagramItem object. This will send information to my_end_item

      :type: DiagramItem

   .. attribute:: my_end_item

      End DiagramItem object. This will get information from my_start_item

      :type: DiagramItem

   .. attribute:: my_color

      Color

      :type: QtCore colour


   .. py:method:: boundingRect()

      Bounding Rectangle.

      Overloaded bounding rectangle. This is necessary to ensure that the
      line and arrowhead are cleaned properly when moving.

      :returns: **tmp**
      :rtype: QtCore.QRectF



   .. py:method:: paint(painter, option, widget=None)

      Overloaded paint method.

      :param painter:
      :type painter: QPainter
      :param option:
      :type option: QStyleOptionGraphicsItem
      :param widget:
      :type widget: QWidget, optional



.. py:class:: DiagramItem(diagram_type, context_menu, my_class, parent)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsPolygonItem`


   Diagram Item.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional
   :param diagram_type: string denoting the diagram type. Can be 'StartEnd', 'Conditional' or
                        'Step'
   :type diagram_type: str
   :param context_menu: Dictionary of context menu options
   :type context_menu: dict
   :param my_class: Class that the diagram item is linked to.
   :type my_class: object

   .. attribute:: arrows

      list of Arrow objects

      :type: list

   .. attribute:: diagram_type

      string denoting the diagram type. Can be 'StartEnd', 'Conditional' or
      'Step'

      :type: str

   .. attribute:: context_menu

      Dictionary of context menu options

      :type: dict

   .. attribute:: my_class

      Class that the diagram item is linked to.

      :type: object

   .. attribute:: text_item

      Text label associated with item.

      :type: None or QtWidgets.QGraphicsTextItem

   .. attribute:: my_class_name

      Class name being referenced

      :type: str


   .. py:method:: add_arrow(arrow)

      Add Arrow.

      :param arrow: Arrow object to add.
      :type arrow: Arrow



   .. py:method:: update_indata()

      Routine to add datasets.



   .. py:method:: contextMenuEvent(event)

      Overloaded context menu event.

      :param event:
      :type event: N/A



   .. py:method:: mouseDoubleClickEvent(event)

      Mouse double click event.

      This event is used to activate an item. It does this by calling the
      settings() method of the item. The event also changes the colour of the
      item to reflect whether it is busy working.



   .. py:method:: remove_arrow(arrow)

      Remove a single Arrow.

      :param arrow: Arrow object to remove.
      :type arrow: Arrow



   .. py:method:: remove_arrows()

      Remove Arrows. Uses the remove_arrow method.



   .. py:method:: settings(nodialog=False)

      Routine Settings.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: **iflag** -- Returns a boolean reflecting success of the my_class.settings()
                method.
      :rtype: bool



.. py:class:: DiagramScene(item_menu, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsScene`


   Diagram Scene.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional
   :param item_menu: Item menu.
   :type item_menu: QtWidgets.QMenu


   .. py:method:: mousePressEvent(mouse_event)

      Overloaded Mouse Press Event.

      :param mouse_event: mouse event.
      :type mouse_event: QGraphicsSceneMouseEvent



   .. py:method:: selected_item_info()

      Display info about selected item.



   .. py:method:: mouseMoveEvent(mouse_event)

      Overloaded Mouse Move Event.

      :param mouse_event: mouse event.
      :type mouse_event: QGraphicsSceneMouseEvent



   .. py:method:: mouseReleaseEvent(mouse_event)

      Overloaded Mouse Release Event.

      :param mouse_event: mouse event.
      :type mouse_event: QGraphicsSceneMouseEvent



.. py:class:: MainWidget(nocgs=True)

   Bases: :py:obj:`PySide6.QtWidgets.QMainWindow`


   Widget class to call the main interface.

   .. attribute:: pdlg



      :type: list

   .. attribute:: context_menu



      :type: dictionary


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: add_to_context(txt)

      Add to a context menu.

      Each dataset type which PyGMI uses can have its own context menu. This
      method allows for the definition of each group of context menu items
      under a user defined text label.

      :param txt: Label for a group of context menu items
      :type txt: str



   .. py:method:: bring_to_front()

      Bring the selected item to front.



   .. py:method:: clearprocesslog()

      Clear the process log.



   .. py:method:: delete_item()

      Delete the selected item from main interface.



   .. py:method:: keyPressEvent(event)

      Intercept key press for custom key presses.

      :param event: Key press event object.
      :type event: QKeyEvent

      :rtype: None.



   .. py:method:: get_indata()

      Get input data from the selected item on the main interface.

      :returns: **idata** -- Input list of PyGMI Data (pygmi.raster.datatypes.Data)
      :rtype: list



   .. py:method:: get_outdata()

      Get output data from the selected item on the main interface.

      :returns: **odata** -- Output list of PyGMI Data (pygmi.raster.datatypes.Data)
      :rtype: list



   .. py:method:: help_docs()

      Help Routine.



   .. py:method:: item_insert(item_type, item_name, class_name, projimport=False, **kwargs)

      Item insert.

      Insert an item on the main interface. The item is an object passed by
      one of the menu.py routines and is one of the algorithms chosen on
      the main PyGMI menu.

      :param item_type: str describing the shape of the graphic used to describe the item.
      :type item_type: str
      :param item_name: str describing the name of the item to be displayed.
      :type item_name: str
      :param class_name: class to be called when double clicking on the item.
      :type class_name: object

      :returns: **item** -- Return a DiagramItem object
      :rtype: DiagramItem



   .. py:method:: launch_context_item(newitem)

      Launch a context menu item, using output data.

      :param newitem: newitem is the class to be called by the context menu item
      :type newitem: custom class



   .. py:method:: launch_context_item_indata(newitem)

      Launch a context menu item, using input data.

      :param newitem: newitem is the class to be called by the context menu item
      :type newitem: custom class



   .. py:method:: linepointer()

      Select line pointer.



   .. py:method:: pointer()

      Select pointer.



   .. py:method:: process_is_active(isactive=True)

      Change process log colour when a process is active.

      :param isactive: boolean variable indicating if a process is active.
      :type isactive: bool, optional



   .. py:method:: load()

      Load project state from JSON file.

      :rtype: None.



   .. py:method:: save()

      Save project state to a JSON file.

      :rtype: None.



   .. py:method:: run()

      Run entire script.

      :rtype: None.



   .. py:method:: send_to_back()

      Send the selected item to the back.



   .. py:method:: showdatainfo(txt)

      Show text in the dataset information panel.

      :param txt: Message to be displayed in the datainfo panel
      :type txt: str



   .. py:method:: showlog(txt, replacelast=False)

      Show text on the process log.

      :param txt: Message to be displayed in the process log
      :type txt: str
      :param replacelast: flag to indicate whether the last row on the log should be
                          overwritten.
      :type replacelast: bool, optional



   .. py:method:: update_pdlg(dlg)

      Clean deleted objects in self.pdlg and appends a new object.

      self.pdlg allows for modeless dialogues to remain in existence until
      they are closed

      :param dlg: Object to be appended to self.pdlg
      :type dlg: object



.. py:class:: Startup(pbarmax)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Class to provide a startup display while PyGMI loads into memory.

   :param pbarmax: Progress bar maximum value.
   :type pbarmax: int


   .. py:method:: update()

      Update the text on the dialog.



.. py:function:: main(nocgs=False)

   Entry point for the PyGMI software.


