# -----------------------------------------------------------------------------
# Name:        main.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2013 Council for Geoscience
# Licence:     GPL-3.0
#
# This file is part of PyGMI
#
# PyGMI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyGMI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
"""
Main module for PyGMI.

This module comprises a series of classes which are responsible for the primary
interface to the software. Credit must go to PyQt's examples, upon which some
of this was originally based.

Although the main interface is controlled here, the content of the menus and
routines is not. The menus and corresponding classes are found within the
pygmi packages.

"""

import sys
import os
import pkgutil
import math
import importlib
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import pygmi
import pygmi.menu_default as menu_default
import pygmi.misc as misc


class Arrow(QtWidgets.QGraphicsLineItem):
    """
    Class responsible for drawing arrows on the main interface.

    Attributes
    ----------
    arrow_head : QPolygonF
        Arrow head polygon.
    my_start_item : DiagramItem
        Starting DiagramItem object. This will send information to my_end_item
    my_end_item : DiagramItem
        End DiagramItem object. This will get information from my_start_item
    my_color : QtCore color (default is QtCore.Qt.black)
        Color
    """

    def __init__(self, start_item, end_item, parent=None):
        super().__init__(parent)

        self.arrow_head = QtGui.QPolygonF()

        self.my_start_item = start_item
        self.my_end_item = end_item
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.my_color = QtCore.Qt.black
        self.setPen(QtGui.QPen(self.my_color, 2, QtCore.Qt.SolidLine,
                               QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

    def boundingRect(self):
        """
        Bounding Rectangle.

        Overloaded bounding rectangle. This is necessary to ensure that the
        line and arrowhead are cleaned properly when moving.

        Returns
        -------
        tmp : QtCore.QRectF
        """
        extra = (self.pen().width() + 20) / 2.0
        p1 = self.line().p1()
        p2 = self.line().p2()
        tmp = QtCore.QRectF(p1, QtCore.QSizeF(p2.x()-p1.x(), p2.y()-p1.y()))

        return tmp.normalized().adjusted(-extra, -extra, extra, extra)

    def paint(self, painter, option, widget=None):
        """
        Overloaded paint method.

        Parameters
        ----------
        painter : QPainter
        option : QStyleOptionGraphicsItem
        widget : QWidget, optional
        """
        pi = math.pi
        if self.my_start_item.collidesWithItem(self.my_end_item):
            return

        my_pen = self.pen()
        my_pen.setColor(self.my_color)
        arrow_size = 10.0
        painter.setPen(my_pen)
        painter.setBrush(self.my_color)

        x0, y0 = np.mean(self.my_start_item.np_poly, 0)
        my_start_off = QtCore.QPointF(x0, y0)
        x1, y1 = np.mean(self.my_end_item.np_poly, 0)
        my_end_off = QtCore.QPointF(x1, y1)

        center_line = QtCore.QLineF(self.my_start_item.pos() + my_start_off,
                                    self.my_end_item.pos() + my_end_off)
        end_polygon = self.my_end_item.polygon()
        p1 = end_polygon.first() + self.my_end_item.pos()

        intersect_point = QtCore.QPointF()
        for i in end_polygon:
            p2 = i + self.my_end_item.pos()
            poly_line = QtCore.QLineF(p1, p2)
            intersect_type = poly_line.intersect(center_line, intersect_point)
            if intersect_type == QtCore.QLineF.BoundedIntersection:
                break
            p1 = p2

        self.setLine(QtCore.QLineF(intersect_point,
                                   self.my_start_item.pos() + my_start_off))
        line = self.line()

        angle = math.acos(line.dx() / line.length())
        if line.dy() >= 0:
            angle = (math.pi * 2.0) - angle

        arrow_p1 = (line.p1() + QtCore.QPointF(math.sin(angle+pi/3) *
                                               arrow_size,
                                               math.cos(angle+pi/3) *
                                               arrow_size))
        arrow_p2 = (line.p1() + QtCore.QPointF(math.sin(angle+pi-pi/3) *
                                               arrow_size,
                                               math.cos(angle+pi-pi/3) *
                                               arrow_size))

        self.arrow_head.clear()
        for point in [line.p1(), arrow_p1, arrow_p2]:
            self.arrow_head.append(point)

        painter.drawLine(line)
        painter.drawPolygon(self.arrow_head)
        if self.isSelected():
            painter.setPen(QtGui.QPen(self.my_color, 1, QtCore.Qt.DashLine))
            my_line = QtCore.QLineF(line)
            my_line.translate(0, 4.0)
            painter.drawLine(my_line)
            my_line.translate(0, -8.0)
            painter.drawLine(my_line)


class DiagramItem(QtWidgets.QGraphicsPolygonItem):
    """
    Diagram Item.

    Attributes
    ----------
    arrows : list
        list of Arrow objects
    diagram_type : str
        string denoting the diagram type. Can be 'StartEnd', 'Conditional' or
        'Step'
    context_menu = context_menu
    my_class : object
        Class that the diagram item is linked to.
    is_import : bool
        Flags whether my_class is used to import data
    text_item : None
    my_class_name : str
        Class name being referenced
    """

    def __init__(self, diagram_type, context_menu, my_class, parent=None):
        super().__init__(parent)

        self.arrows = []
        self.parent = my_class.parent

        self.diagram_type = diagram_type
        self.context_menu = context_menu
        self.my_class = my_class
        self.is_import = False
        self.text_item = None
        self.my_class_name = ''

        if hasattr(self.my_class, 'arrows'):
            self.my_class.arrows = self.arrows

        path = QtGui.QPainterPath()
        if self.diagram_type == 'StartEnd':
            path.moveTo(200, 50)
            path.arcTo(150, 0, 50, 50, 0, 90)
            path.arcTo(50, 0, 50, 50, 90, 90)
            path.arcTo(50, 50, 50, 50, 180, 90)
            path.arcTo(150, 50, 50, 50, 270, 90)
            path.lineTo(200, 25)
            self.my_polygon = path.toFillPolygon()
        elif self.diagram_type == 'Conditional':
            self.np_poly = np.array([[-100., 0.],
                                     [0., 100.],
                                     [100., 0.],
                                     [0., -100.],
                                     [-100., 0.]])

            my_points = []
            for i in self.np_poly:
                my_points.append(QtCore.QPointF(i[0], i[1]))
            self.my_polygon = QtGui.QPolygonF(my_points)

        elif self.diagram_type == 'Step':
            self.np_poly = np.array([[-100., -100.],
                                     [100., -100.],
                                     [100., 100.],
                                     [-100., 100.],
                                     [-100., -100.]])

            my_points = []
            for i in self.np_poly:
                my_points.append(QtCore.QPointF(i[0], i[1]))
            self.my_polygon = QtGui.QPolygonF(my_points)

        else:
            self.np_poly = np.array([[-120., -80.],
                                     [-70., 80.],
                                     [120., 80.],
                                     [70., -80.],
                                     [-120., -80.]])

            my_points = []
            for i in self.np_poly:
                my_points.append(QtCore.QPointF(i[0], i[1]))
            self.my_polygon = QtGui.QPolygonF(my_points)

        self.setPolygon(self.my_polygon)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)

    def add_arrow(self, arrow):
        """Add Arrow.

        Parameters
        ----------
        arrow : Arrow
            Arrow object to add.
        """
        self.arrows.append(arrow)

    def update_indata(self):
        """Routine to add datasets."""
        data = {}
        for i in self.arrows:
            odata = i.my_start_item.my_class.outdata
            for j in odata:
                if j in data:
                    data[j] = data[j] + odata[j]
                else:
                    data[j] = odata[j]

        self.my_class.indata = data
        if hasattr(self.my_class, 'data_init'):
            self.my_class.data_init()

    def contextMenuEvent(self, event):
        """
        Overloaded context menu event.

        Parameters
        ----------
        event : N/A
        """
        self.scene().clearSelection()
        self.setSelected(True)

        exclude = ['ProfPic', 'GenFPS']

        tmp = self.context_menu['Basic'].actions()
        if 'Raster' in self.my_class.indata:
            tmp += self.context_menu['inRaster'].actions()

        tmplist = list(self.my_class.outdata.keys())
        for i in tmplist:
            if i not in exclude:
                tmp += self.context_menu[i].actions()
            if i == 'ProfPic' and 'Raster' not in tmplist:
                tmp += self.context_menu['Raster'].actions()
        local_menu = QtWidgets.QMenu()
        local_menu.addActions(tmp)
        local_menu.exec_(event.screenPos())

    def mouseDoubleClickEvent(self, event):
        """
        Mouse double click event.

        This event is used to activate an item. It does this by calling the
        settings() method of the item. The event also changes the color of the
        item to reflect whether it is busy working.
        """
        self.setBrush(QtGui.QColor(255, 0, 0, 127))

        temp = self.settings()
        self.parent.scene.selected_item_info()

        if temp is True:
            self.setBrush(QtGui.QColor(0, 255, 0, 127))
        else:
            self.setBrush(self.scene().my_item_color)

    def remove_arrow(self, arrow):
        """
        Remove a single Arrow.

        Parameters
        ----------
        arrow : Arrow
            Arrow object to remove.
        """
        try:
            self.arrows.remove(arrow)
        except ValueError:
            pass

    def remove_arrows(self):
        """Remove Arrows. Uses the remove_arrow method."""
        for arrow in self.arrows[:]:
            arrow.my_start_item.remove_arrow(arrow)
            arrow.my_end_item.remove_arrow(arrow)
            self.scene().removeItem(arrow)

    def settings(self):
        """
        Routine Settings.

        Returns
        -------
        iflag : bool
            Returns a boolean reflecting success of the my_class.settings()
            method.
        """
        if self.is_import is True:
            pass
        elif self.my_class.indata == {} and self.is_import is False:
            QtWidgets.QMessageBox.warning(self.parent, 'Warning',
                                          ' You need to connect data first!',
                                          QtWidgets.QMessageBox.Ok)
            return False

        self.my_class.parent.process_is_active()
        self.my_class.parent.showprocesslog(self.my_class_name+' busy...')
        iflag = self.my_class.settings()
        self.my_class.parent.process_is_active(False)
        self.my_class.parent.showprocesslog(self.my_class_name+' finished!')
        return iflag


class DiagramScene(QtWidgets.QGraphicsScene):
    """Diagram Scene."""

    def __init__(self, item_menu, parent):
        super().__init__(parent)

        self.my_item_menu = item_menu
        self.my_mode = 'MoveItem'
        self.my_item_type = 'Step'
        self.line = None
        self.text_item = None
        self.my_item_color = QtCore.Qt.cyan
        self.my_text_color = QtCore.Qt.black
        self.my_line_color = QtCore.Qt.black
        self.my_font = QtGui.QFont()
        self.parent = parent

    def mousePressEvent(self, mouse_event):
        """
        Overloaded Mouse Press Event.

        Parameters
        ----------
        mouse_event: QGraphicsSceneMouseEvent
            mouse event.
        """
        if mouse_event.button() != QtCore.Qt.LeftButton:
            return
        if self.my_mode == 'InsertLine':
            self.line = QtWidgets.QGraphicsLineItem(
                QtCore.QLineF(mouse_event.scenePos(), mouse_event.scenePos()))
            self.line.setPen(QtGui.QPen(self.my_line_color, 2))
            self.addItem(self.line)

        super().mousePressEvent(mouse_event)

        self.selected_item_info()

# now display the information about the selected data
    def selected_item_info(self):
        """Display info about selected item."""
        tmp = self.selectedItems()
        if not tmp:
            return

        text = ''

        if hasattr(tmp[0].my_class, 'indata'):
            idata = tmp[0].my_class.indata

            for i in idata:
                text += '\nInput ' + i + ' dataset:\n'
                if i in ('Raster'):
                    for j in idata[i]:
                        text += '  '+j.dataid + '\n'

        if hasattr(tmp[0].my_class, 'outdata'):
            odata = tmp[0].my_class.outdata

            for i in odata:
                text += '\nOutput ' + i + ' dataset:\n'
                if i in ('Raster', 'Cluster'):
                    for j in odata[i]:
                        text += '  '+j.dataid + '\n'
                if i == 'Model3D':
                    for j in odata[i][0].lith_list:
                        text += '  '+j + '\n'
                if i == 'MT - EDI':
                    for j in odata[i]:
                        text += '  '+j + '\n'

        self.parent.showdatainfo(text)


    def mouseMoveEvent(self, mouse_event):
        """
        Overloaded Mouse Move Event.

        Parameters
        ----------
        mouse_event: QGraphicsSceneMouseEvent
            mouse event.
        """
        if self.my_mode == 'InsertLine' and self.line:
            new_line = QtCore.QLineF(self.line.line().p1(),
                                     mouse_event.scenePos())
            self.line.setLine(new_line)
        elif self.my_mode == 'MoveItem':
            super().mouseMoveEvent(mouse_event)

    def mouseReleaseEvent(self, mouse_event):
        """
        Overloaded Mouse Release Event.

        Parameters
        ----------
        mouse_event: QGraphicsSceneMouseEvent
            mouse event.
        """
        if self.line and self.my_mode == 'InsertLine':
            start_items = self.items(self.line.line().p1())
            if start_items and start_items[0] == self.line:
                start_items.pop(0)
            end_items = self.items(self.line.line().p2())
            if end_items and end_items[0] == self.line:
                end_items.pop(0)

            self.removeItem(self.line)
            self.line = None

            if (start_items and end_items and
                    isinstance(start_items[-1], DiagramItem) and
                    isinstance(end_items[-1], DiagramItem) and
                    start_items[-1] != end_items[-1]):
                start_item = start_items[-1]
                end_item = end_items[-1]
                arrow = Arrow(start_item, end_item)
                start_item.add_arrow(arrow)
                end_item.add_arrow(arrow)
                arrow.setZValue(-1000.0)
                self.addItem(arrow)
                end_item.update_indata()

        self.line = None
        self.my_mode = 'MoveItem'
        self.parent.action_pointer.setChecked(True)
        super().mouseReleaseEvent(mouse_event)


class MainWidget(QtWidgets.QMainWindow):
    """
    Widget class to call the main interface.

    Attributes
    ----------
    pdlg : list
    context_menu : dictionary
    """

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        ipth = os.path.dirname(menu_default.__file__)+r'/images/'

        self.__version__ = pygmi.__version__
        self.pdlg = []
        self.context_menu = {}
        self.add_to_context('Basic')

        self.menubar = QtWidgets.QMenuBar()

        self.statusbar = QtWidgets.QStatusBar()
        self.toolbar = QtWidgets.QToolBar()

        self.centralwidget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.centralwidget)
        self.graphics_view = QtWidgets.QGraphicsView()
        self.textbrowser_datainfo = QtWidgets.QTextBrowser()
        self.textbrowser_processlog = QtWidgets.QTextBrowser()
        self.pbar = misc.ProgressBar()

        self.action_help = QtWidgets.QAction(self)
        self.action_delete = QtWidgets.QAction(self)
        self.action_bring_to_front = QtWidgets.QAction(self)
        self.action_send_to_back = QtWidgets.QAction(self)
        self.action_pointer = QtWidgets.QAction(self)
        self.action_linepointer = QtWidgets.QAction(self)
        self.actiongroup_pointer = QtWidgets.QActionGroup(self)
        self.actiongroup_pointer.addAction(self.action_pointer)
        self.actiongroup_pointer.addAction(self.action_linepointer)
        self.action_pointer.setCheckable(True)
        self.action_linepointer.setCheckable(True)
        self.action_pointer.setChecked(True)

        self.action_delete.setIcon(QtGui.QIcon(ipth+'delete.png'))
        self.action_bring_to_front.setIcon(
            QtGui.QIcon(ipth+'bringtofront.png'))
        self.action_send_to_back.setIcon(QtGui.QIcon(ipth+'sendtoback.png'))
        self.action_pointer.setIcon(QtGui.QIcon(ipth+'pointer.png'))
        self.action_linepointer.setIcon(QtGui.QIcon(ipth+'linepointer.png'))
        self.action_help.setIcon(QtGui.QIcon(ipth+'help.png'))

        self.setWindowIcon(QtGui.QIcon(ipth+'logo256.ico'))
        self.setupui()

        menus = []
        for _, modname, _ in pkgutil.walk_packages(
                path=pygmi.__path__, prefix=pygmi.__name__+'.',
                onerror=lambda x: None):
            menus.append(modname)

        menus.pop(menus.index('pygmi.rsense.menu'))
        menus.pop(menus.index('pygmi.mt.menu'))
        raster_menu = menus.pop(menus.index('pygmi.raster.menu'))
        vector_menu = menus.pop(menus.index('pygmi.vector.menu'))
        menus = [raster_menu, vector_menu]+menus

        menus = [i for i in menus if 'menu' in i[-5:]]
        start = Startup(len(menus)+1)
        start.update()

        menuimports = []
        for i in menus:
            if i == 'pygmi.__pycache__.menu':
                continue
            start.update()
            menuimports.append(importlib.import_module(i))
        start.close()

        self.menus = []
        self.menus.append(menu_default.FileMenu(self))
        for i in menuimports:
            self.menus.append(i.MenuWidget(self))
        self.menus.append(menu_default.HelpMenu(self))

        self.scene = DiagramScene(self.context_menu['Basic'], self)

        self.view = self.graphics_view
        self.view.setScene(self.scene)

# Menus
        self.action_pointer.triggered.connect(self.pointer)
        self.action_linepointer.triggered.connect(self.linepointer)
        self.action_delete.triggered.connect(self.delete_item)
        self.action_bring_to_front.triggered.connect(self.bring_to_front)
        self.action_send_to_back.triggered.connect(self.send_to_back)
        self.action_help.triggered.connect(self.help_docs)

# Start of Functions
    def setupui(self):
        """Set up UI."""
        self.resize(800, 600)
        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Expanding)
        sizepolicy.setHorizontalStretch(0)
        sizepolicy.setVerticalStretch(0)
        sizepolicy.setHeightForWidth(
            self.graphics_view.sizePolicy().hasHeightForWidth())

        self.graphics_view.setSizePolicy(sizepolicy)
        self.graphics_view.setTransformationAnchor(
            QtWidgets.QGraphicsView.AnchorUnderMouse)

        self.textbrowser_datainfo.setSizePolicy(sizepolicy)
        self.textbrowser_processlog.setSizePolicy(sizepolicy)
        self.textbrowser_processlog.setStyleSheet(
            '* { background-color: rgb(255, 255, 255); }')

        self.grid_layout.addWidget(self.graphics_view, 0, 0, 4, 2)
        self.grid_layout.addWidget(self.textbrowser_datainfo, 1, 2, 1, 1)
        self.grid_layout.addWidget(self.textbrowser_processlog, 3, 2, 1, 1)
        self.grid_layout.addWidget(self.pbar, 5, 0, 1, 3)

        label = QtWidgets.QLabel('Dataset Information:')
        label_2 = QtWidgets.QLabel('Process Log:')
        self.grid_layout.addWidget(label, 0, 2, 1, 1)
        self.grid_layout.addWidget(label_2, 2, 2, 1, 1)

        self.setCentralWidget(self.centralwidget)
        self.setMenuBar(self.menubar)
        self.setStatusBar(self.statusbar)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        self.toolbar.addAction(self.action_delete)
        self.toolbar.addAction(self.action_bring_to_front)
        self.toolbar.addAction(self.action_send_to_back)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_pointer)
        self.toolbar.addAction(self.action_linepointer)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_help)

        self.setWindowTitle(
            'PyGMI - Python Geoscience Modelling and Interpretation')
        self.action_delete.setText('Delete')
        self.action_bring_to_front.setText('Bring to Front')
        self.action_send_to_back.setText('Send to Back')
        self.action_pointer.setText('Pointer')
        self.action_linepointer.setText('LinePointer')

        item_menu = self.context_menu['Basic']
        item_menu.addAction(self.action_delete)
        item_menu.addAction(self.action_bring_to_front)
        item_menu.addAction(self.action_send_to_back)

    def add_to_context(self, txt):
        """
        Add to a context menu.

        Each dataset type which PyGMI uses can have its own context menu. This
        method allows for the definition of each group of context menu items
        under a user defined text label.

        Parameters
        ----------
        txt : str
            Label for a group of context menu items
        """
        if self.context_menu.get(txt) is not None:
            return
        self.context_menu[txt] = QtWidgets.QMenu()

    def bring_to_front(self):
        """Bring the selected item to front."""
        if not self.scene.selectedItems():
            return

        selected_item = self.scene.selectedItems()[0]
        overlap_items = selected_item.collidingItems()

        zvalue = 0
        for item in overlap_items:
            if item.zValue() >= zvalue and isinstance(item, DiagramItem):
                zvalue = item.zValue() + 0.1
        selected_item.setZValue(zvalue)

    def clearprocesslog(self):
        """Clear the process log."""
        self.textbrowser_processlog.setPlainText('')

    def delete_item(self):
        """Delete the selected item from main interface."""
        for item in self.scene.selectedItems():
            if isinstance(item, DiagramItem):
                item.remove_arrows()
            self.scene.removeItem(item)

        for item in self.scene.items():
            if isinstance(item, DiagramItem):
                item.update_indata()
                if item.my_class.indata == {} and item.is_import is False:
                    item.setBrush(self.scene.my_item_color)

    def get_indata(self):
        """
        Get input data from the selected item on the main interface.

        Returns
        -------
        idata : list
            Input list of PyGMI dataset
        """
        idata = []
        for item in self.scene.selectedItems():
            if isinstance(item, DiagramItem):
                idata.append(item.my_class.indata)
        return idata

    def get_outdata(self):
        """
        Get output data from the selected item on the main interface.

        Returns
        -------
        odata : list
            Output list of PyGMI dataset
        """
        odata = []
        for item in self.scene.selectedItems():
            if isinstance(item, DiagramItem):
                odata.append(item.my_class.outdata)
        return odata

    def help_docs(self):
        """Help Routine."""
        menu_default.HelpDocs(self, 'pygmi.main')

    def item_insert(self, item_type, item_name, class_name):
        """
        Item insert.

        Insert an item on the main interface. The item is an object passed by
        one of the menu.py routines and is one of the algorithims chosen on
        the main PyGMI menu.

        Parameters
        ----------
        item_type : str
            str describing the shape of the graphic used to describe the item.
        item_name : str
            str describing the name of the item to be displayed.
        class_name : object
            class to be called when double clicking on the item.

        Returns
        -------
        item : DiagramItem
            Return a DiagramItem object
        """
        item = DiagramItem(item_type, self.scene.my_item_menu, class_name)
        item_color = self.scene.my_item_color

        item.my_class_name = item_name.replace('\n', ' ')

        if 'Import' in item_name:
            item.is_import = True
            iflag = item.settings()
            if iflag is False:
                return None
            ifile = os.path.basename(item.my_class.ifile)
            if len(ifile) > len(item_name):
                ifile = ifile[:len(item_name)]+'\n'+ifile[len(item_name):]
            if len(ifile) > 2*len(item_name):
                ifile = ifile[:2*len(item_name)]+'...\n'
            item_name += ':\n'+ifile
            item_color = QtGui.QColor(0, 255, 0, 127)

# Do text first, since this determines size of polygon
        text_item = QtWidgets.QGraphicsTextItem()
        text_item.setPlainText(item_name)
        text_item.setFont(self.scene.my_font)
        text_item.setZValue(1000.0)
        text_item.setDefaultTextColor(self.scene.my_text_color)

# Rectangle for text label
        rect_item = QtWidgets.QGraphicsRectItem(text_item.boundingRect())
        rect_item.setZValue(500.0)
        rect_item.setBrush(self.scene.my_item_color)

# Actual polygon item
        text_width = text_item.boundingRect().width()
        item.np_poly *= 1.5*text_width/item.np_poly[:, 0].ptp()
        item.np_poly[:, 0] += (text_item.boundingRect().left() -
                               item.np_poly[:, 0].min() - text_width/4)
        item.np_poly[:, 1] += text_item.boundingRect().height()/2

        my_points = []
        for i in item.np_poly:
            my_points.append(QtCore.QPointF(i[0], i[1]))
        item.my_polygon = QtGui.QPolygonF(my_points)

        item.setPolygon(item.my_polygon)
        item.setBrush(item_color)

# Menu Stuff
        item.context_menu = self.context_menu

# Add item to scene and merge
        self.scene.addItem(item)

        xxyy = self.view.mapToScene(self.view.width()/2.,
                                    self.view.height()/2.)
        item.setPos(xxyy)

        text_item.setParentItem(item)
        rect_item.setParentItem(item)

# Enable moving
        self.scene.my_mode = 'MoveItem'
        return item

    def launch_context_item(self, newitem):
        """
        Launch a context menu item, using output data.

        Parameters
        ----------
        newitem : custom class
            newitem is the class to be called by the context menu item
        """
        outdata = self.get_outdata()

        for odata in outdata:
            if odata is not None and odata != {}:
                dlg = newitem(self)
                dlg.indata = odata
                dlg.run()
                self.update_pdlg(dlg)

    def launch_context_item_indata(self, newitem):
        """
        Launch a context menu item, using input data.

        Parameters
        ----------
        newitem : custom class
            newitem is the class to be called by the context menu item
        """
        indata = self.get_indata()

        for idata in indata:
            if idata is not None and idata != {}:
                dlg = newitem(self)
                dlg.indata = idata
                dlg.run()
                self.update_pdlg(dlg)

    def linepointer(self):
        """Select line pointer."""
        self.scene.my_mode = 'InsertLine'

    def pointer(self):
        """Select pointer."""
        self.scene.my_mode = 'MoveItem'

    def process_is_active(self, isactive=True):
        """
        Change process log color when a process is active.

        Parameters
        ----------
        isactive : bool, optional
            boolean variable indicating if a process is active.
        """
        if isactive:
            self.textbrowser_processlog.setStyleSheet(
                '* { background-color: rgba(255, 0, 0, 127); }')
            self.pbar.setValue(0)
        else:
            self.textbrowser_processlog.setStyleSheet(
                '* { background-color: rgb(255, 255, 255); }')

    def send_to_back(self):
        """Send the selected item to the back."""
        if not self.scene.selectedItems():
            return

        selected_item = self.scene.selectedItems()[0]
        overlap_items = selected_item.collidingItems()

        zvalue = 0
        for item in overlap_items:
            if item.zValue() <= zvalue and isinstance(item, DiagramItem):
                zvalue = item.zValue() - 0.1
        selected_item.setZValue(zvalue)

    def showdatainfo(self, txt):
        """
        Show text in the dataset information panel.

        Parameters
        ----------
        txt : str
            Message to be displayed in the datainfo panel
        """
        self.textbrowser_datainfo.setPlainText(txt)
        tmp = self.textbrowser_datainfo.verticalScrollBar()
        tmp.setValue(tmp.maximumHeight())
        self.repaint()

    def showprocesslog(self, txt, replacelast=False):
        """
        Show text on the process log.

        Parameters
        ----------
        txt : str
            Message to be displayed in the process log
        replacelast : bool, optional
            flag to indicate whether the last row on the log should be
            overwritten.
        """
        self.showtext(self.textbrowser_processlog, txt, replacelast)
        QtWidgets.QApplication.processEvents()

    def showtext(self, txtobj, txt, replacelast=False):
        """
        Show text on the text panel.

        Parameters
        ----------
        txt : str
            Message to be displayed in the text panel
        replacelast : bool, optional
            flag to indicate whether the last row on the log should be
            overwritten.
        """
        txtmsg = str(txtobj.toPlainText())
        if replacelast is True:
            txtmsg = txtmsg[:txtmsg.rfind('\n')]
            txtmsg = txtmsg[:txtmsg.rfind('\n')]
            txtmsg += '\n'
        txtmsg += txt + '\n'
        txtobj.setPlainText(txtmsg)
        tmp = txtobj.verticalScrollBar()
        tmp.setValue(tmp.maximumHeight())
        self.repaint()

    def update_pdlg(self, dlg):
        """
        Clean deleted objects in self.pdlg and appends a new object.

        self.pdlg allows for modeless dialogs to remain in existance until they
        are closed

        Parameters
        ----------
        dlg : object
            Object to be appended to self.pdlg
        """
        for i in range(len(self.pdlg)-1, -1, -1):
            try:
                if not self.pdlg[i].isVisible():
                    self.pdlg.pop(i)
            except RuntimeError:
                self.pdlg.pop(i)
            except AttributeError:
                self.pdlg.pop(i)

        self.pdlg.append(dlg)


class Startup(QtWidgets.QDialog):
    """Class to provide a startup display while PyGMI loads into memory."""

    def __init__(self, pbarmax, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.ToolTip)

        self.gridlayout_main = QtWidgets.QVBoxLayout(self)
        self.label_info = QtWidgets.QLabel(self)
        self.label_pic = QtWidgets.QLabel(self)
        self.label_pic.setPixmap(QtGui.QPixmap(pygmi.__path__[0] +
                                               r'/images/logo256.ico'))
        self.label_info.setScaledContents(True)
        self.pbar = QtWidgets.QProgressBar(self)

        labeltext = "<font color='red'>Py</font><font color='blue'>GMI</font>"

        fnt = QtGui.QFont('Arial', 72, QtGui.QFont.Bold)
        self.label_info.setFont(fnt)
        self.label_info.setText(labeltext)
        self.gridlayout_main.addWidget(self.label_info)
        self.gridlayout_main.addWidget(self.label_pic)

        self.pbar.setMaximum(pbarmax - 1)
        self.gridlayout_main.addWidget(self.pbar)

        self.open()

    def update(self):
        """Update the text on the dialog."""
        self.pbar.setValue(self.pbar.value() + 1)
        QtWidgets.QApplication.processEvents()


def main():
    """Entry point for the PyGMI software."""
    app = QtWidgets.QApplication(sys.argv)

    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()

    wid = MainWidget()
    wid.resize(width*0.75, height*0.75)

    wid.setWindowState(wid.windowState() & ~QtCore.Qt.WindowMinimized |
                       QtCore.Qt.WindowActive)

    # this will activate the window
    wid.show()
    wid.activateWindow()

    try:
        __IPYTHON__
    except NameError:
        sys.exit(app.exec_())
    else:
        app.exec_()


if __name__ == "__main__":
    main()
