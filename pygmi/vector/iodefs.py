# -----------------------------------------------------------------------------
# Name:        iodefs.py (part of PyGMI)
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
"""Import Data."""

import os
import glob
import re
from io import StringIO

from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona

from pygmi import menu_default
from pygmi.raster.dataprep import GroupProj
from pygmi.misc import BasicModule, ContextModule
from pygmi.vector.dataprep import maptobounds


class ColumnSelect(BasicModule):
    """A combobox to select vector columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Column Selection')

        self.vbl = QtWidgets.QVBoxLayout()
        self.setLayout(self.vbl)

        self.lw_1 = QtWidgets.QListWidget()
        self.lw_1.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.vbl.addWidget(self.lw_1)

        self.buttonbox = QtWidgets.QDialogButtonBox()
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setCenterButtons(True)
        self.buttonbox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        self.vbl.addWidget(self.buttonbox)

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        data = self.indata['Vector'][0]
        tmp = list(data.columns)
        tmp = [i for i in tmp if i != 'geometry']

        self.lw_1.addItems(tmp)

        if not tmp:
            return False

        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return False

        atmp = [i.text() for i in self.lw_1.selectedItems()]
        if 'geometry' in data.columns:
            atmp.append('geometry')

        data = data.loc[:, atmp]

        self.outdata['Vector'] = [data]

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        # self.saveobj(self.ifile)


class ImportXYZ(BasicModule):
    """
    Import XYZ Data.

    This class imports tabular data.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filt = ''
        self.is_import = True

        self.cmb_xchan = QtWidgets.QComboBox()
        self.cmb_ychan = QtWidgets.QComboBox()
        self.le_nodata = QtWidgets.QLineEdit('99999')
        self.proj = GroupProj('Input Projection')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.vector.iodefs.'
                                           'importxyzdata')
        lbl_xchan = QtWidgets.QLabel('X Channel:')
        lbl_ychan = QtWidgets.QLabel('Y Channel:')
        lbl_nodata = QtWidgets.QLabel('Nodata Value:')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.cmb_xchan.setSizeAdjustPolicy(3)
        self.cmb_ychan.setSizeAdjustPolicy(3)
        self.cmb_xchan.setMinimumSize = 10
        self.cmb_ychan.setMinimumSize = 10

        self.setWindowTitle(r'Import XYZ Data')

        gl_main.addWidget(lbl_xchan, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_xchan, 0, 1, 1, 1)

        gl_main.addWidget(lbl_ychan, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_ychan, 1, 1, 1, 1)

        gl_main.addWidget(lbl_nodata, 2, 0, 1, 1)
        gl_main.addWidget(self.le_nodata, 2, 1, 1, 1)

        gl_main.addWidget(helpdocs, 5, 0, 1, 1)
        gl_main.addWidget(buttonbox, 5, 1, 1, 3)

        gl_main.addWidget(self.proj, 3, 0, 1, 4)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if not nodialog:
            ext = ('Common formats (*.xlsx *.csv);;'
                   'Excel (*.xlsx);;'
                   'Comma Delimited (*.csv);;'
                   'Geosoft XYZ (*.xyz);;'
                   'ASCII XYZ (*.xyz);;'
                   'Space Delimited (*.txt);;'
                   'Tab Delimited (*.txt);;'
                   'Intrepid Database (*..DIR)')

            self.ifile, self.filt = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
            if self.ifile == '':
                return False

        if self.filt == 'Geosoft XYZ (*.xyz)':
            gdf = self.get_GXYZ()

        elif '.xlsx' in self.ifile:
            gdf = self.get_excel()
        elif self.filt == 'ASCII XYZ (*.xyz)':
            gdf = self.get_delimited(' ')
        elif '.csv' in self.ifile:
            gdf = self.get_delimited(',')
        elif self.filt == 'Tab Delimited (*.txt)':
            gdf = self.get_delimited('\t')
        elif self.filt == 'Space Delimited (*.txt)':
            gdf = self.get_delimited(' ')
        elif self.filt == 'Intrepid Database (*..DIR)':
            gdf = get_intrepid(self.ifile, self.showlog, self.piter)
            self.le_nodata.setDisabled(True)
        else:
            return False

        if gdf is None:
            return False

        self.proj.set_current('None')

        self.cmb_xchan.clear()
        self.cmb_ychan.clear()

        xind = -1
        yind = -1

        ltmp = gdf.columns.str.lower()

        exactpairs = [['lon', 'lat'],
                      ['long', 'lat'],
                      ['longitude', 'latitude'],
                      ['x', 'y'],
                      ['e', 'n']]

        for i, j in exactpairs:
            if i in ltmp and j in ltmp:
                xind = ltmp.get_loc(i)
                yind = ltmp.get_loc(j)
                if 'lon' in i:
                    self.proj.cmb_datum.setCurrentIndex(1)
                break

        if xind == -1:
            ltmp = ltmp.values
            # Check for flexible matches
            for i, tmp in enumerate(ltmp):
                tmpl = tmp.lower()
                if 'lon' in tmpl or 'x' in tmpl or 'east' in tmpl:
                    xind = i
                if 'lat' in tmpl or 'y' in tmpl or 'north' in tmpl:
                    yind = i

        if xind == -1:
            xind = 0
            yind = 1

        self.cmb_xchan.addItems(gdf.columns.values)
        self.cmb_ychan.addItems(gdf.columns.values)

        self.cmb_xchan.setCurrentIndex(xind)
        self.cmb_ychan.setCurrentIndex(yind)

        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return tmp

        try:
            nodata = float(self.le_nodata.text())
        except ValueError:
            self.showlog('Nodata Value error - abandoning import')
            return False

        xcol = self.cmb_xchan.currentText()
        ycol = self.cmb_ychan.currentText()

        x = gdf[xcol]
        y = gdf[ycol]

        if x.dtype == 'O' or y.dtype == 'O':
            self.showlog('Error: You have text in your coordinates.')
            return False

        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(x, y))

        gdf['line'] = gdf['line'].astype(str)

        if self.le_nodata.isEnabled():
            gdf = gdf.replace(nodata, np.nan)

        if self.proj.wkt != '':
            gdf = gdf.set_crs(self.proj.wkt)

        gdf.attrs['source'] = os.path.basename(self.ifile)
        self.outdata['Vector'] = [gdf]

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)
        self.saveobj(self.filt)
        self.saveobj(self.cmb_xchan)
        self.saveobj(self.cmb_ychan)
        self.saveobj(self.le_nodata)

    def get_GXYZ(self):
        """
        Get Geosoft XYZ.

        Returns
        -------
        df : DataFrame
            Pandas dataframe.

        """
        df = get_GXYZ(self.ifile, self.showlog, self.piter)

        return df

    def get_delimited(self, delimiter=','):
        """
        Get a delimited file.

        Parameters
        ----------
        delimiter : str, optional
            Delimiter type. The default is ','.

        Returns
        -------
        gdf : Dataframe
            Pandas dataframe.

        """
        try:
            gdf = pd.read_csv(self.ifile, delimiter=delimiter,
                              index_col=False)
        except:
            self.showlog('Error reading file.')
            return None

        gdf.columns = gdf.columns.str.lower()

        if 'line' not in gdf.columns:
            gdf['line'] = 'None'

        return gdf

    def get_excel(self):
        """
        Get an Excel spreadsheet.

        Returns
        -------
        gdf : Dataframe
            Pandas dataframe.

        """
        gdf = pd.read_excel(self.ifile)

        gdf.columns = gdf.columns.str.lower()

        if 'line' not in gdf.columns:
            gdf['line'] = 'None'

        return gdf


class ExportXYZ(ContextModule):
    """Export XYZ Data."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        """
        Run routine.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        self.parent.process_is_active(True)

        if 'Vector' not in self.indata:
            self.showlog('Error: You need to have line data first!')
            self.parent.process_is_active(False)
            return False

        data = self.indata['Vector'][0]
        if data.geom_type.iloc[0] != 'Point':
            self.showlog('No point type data.')
            self.parent.process_is_active(False)
            return False

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'csv (*.csv);; Excel (*.xlsx)')

        if filename == '':
            self.parent.process_is_active(False)
            return False

        self.showlog('Export busy...')

        os.chdir(os.path.dirname(filename))

        filt = (data.columns != 'geometry')
        cols = list(data.columns[filt])

        # from https://stackoverflow.com/questions/64695352/pandas-to-csv-
        # progress-bar-with-tqdm
        chunks = np.array_split(data.index, 100)  # split into 100 chunks
        chunks = [i for i in chunks if i.size > 0]

        if filename[-3:] == 'csv':
            for chunck, subset in enumerate(self.piter(chunks)):
                if chunck == 0:  # first row
                    data.loc[subset].to_csv(filename, mode='w', index=False,
                                            columns=cols)
                else:
                    data.loc[subset].to_csv(filename, header=None, mode='a',
                                            index=False, columns=cols)
        else:

            if data.shape[0] > 1048576:
                self.showlog('Your data has too many rows. Truncating it to '
                             '1,048,576 rows')
                data2 = data.iloc[:1048576]
            else:
                data2 = data

            data2.to_excel(filename, index=False, columns=cols)

        self.parent.process_is_active(False)

        self.showlog('Export completed')

        return True


class ExportVector(ContextModule):
    """Export Vector Data."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        """
        Run routine.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        self.parent.process_is_active(True)

        if 'Vector' not in self.indata:
            self.showlog('Error: You need to have vector data first!')
            self.parent.process_is_active(False)
            return False

        filename, filt = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'shp (*.shp);;GeoJSON (*.geojson);;'
            'GeoPackage (*.gpkg)')

        if filename == '':
            self.parent.process_is_active(False)
            return False

        self.showlog('Export busy...')

        os.chdir(os.path.dirname(filename))
        data = self.indata['Vector'][0]

        if filt == 'GeoPackage (*.gpkg)' and 'fid' in data.columns.str.lower():
            for i in data.columns:  # don't know case of fid column name
                if i.lower() == 'fid':
                    data['fid_original'] = data[i]
                    data.drop(i, axis=1, inplace=True)
                    break

        if filt == 'shp (*.shp)':
            test = [i for i in data.columns if len(i) > 10]
            if test:
                self.showlog('You have columns with more than 10 characters. '
                             'They will be renamed but could still cause '
                             'problems.')

        chunks = np.array_split(data.index, 100)  # split into 100 chunks
        chunks = [i for i in chunks if i.size > 0]

        try:
            for chunck, subset in enumerate(self.piter(chunks)):
                if chunck == 0:  # first row
                    data.loc[subset].to_file(filename, engine='pyogrio')
                else:
                    data.loc[subset].to_file(filename, engine='pyogrio',
                                             append=True)
        except RuntimeError as e:
            self.showlog(str(e))
            self.showlog('Export aborted.')
            self.parent.process_is_active(False)
            return False

        self.showlog('Export completed')
        self.parent.process_is_active(False)

        return True


class ImportVector(BasicModule):
    """Import Vector Data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_import = True
        self.crs = None

        self.cmb_bounds = QtWidgets.QComboBox()
        self.le_sfile = QtWidgets.QLineEdit('')
        self.le_xmin = QtWidgets.QLineEdit('0.0')
        self.le_xmax = QtWidgets.QLineEdit('1.0')
        self.le_ymin = QtWidgets.QLineEdit('0.0')
        self.le_ymax = QtWidgets.QLineEdit('1.0')
        self.le_mapsheet = QtWidgets.QLineEdit('2918AA')
        self.lbl_xmin = QtWidgets.QLabel('West:')
        self.lbl_xmax = QtWidgets.QLabel('East:')
        self.lbl_ymin = QtWidgets.QLabel('South:')
        self.lbl_ymax = QtWidgets.QLabel('North:')
        self.lbl_mapsheet = QtWidgets.QLabel('Mapsheet:')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        pb_sfile = QtWidgets.QPushButton(' Filename')

        pixmapi = QtWidgets.QStyle.SP_DialogOpenButton
        icon = self.style().standardIcon(pixmapi)
        pb_sfile.setIcon(icon)
        pb_sfile.setStyleSheet('text-align:left;')

        self.setWindowTitle('Import Vector Data')

        self.cmb_bounds.addItems(['None', 'Manual', 'SA Mapsheet'])

        gl_1 = QtWidgets.QGridLayout(self)

        gl_1.addWidget(pb_sfile, 1, 0, 1, 1)
        gl_1.addWidget(self.le_sfile, 1, 1, 1, 1)
        gl_1.addWidget(QtWidgets.QLabel('Bounds:'), 2, 0, 1, 1)
        gl_1.addWidget(self.cmb_bounds, 2, 1, 1, 1)
        gl_1.addWidget(self.lbl_xmin, 3, 0, 1, 1)
        gl_1.addWidget(self.le_xmin, 3, 1, 1, 1)
        gl_1.addWidget(self.lbl_xmax, 4, 0, 1, 1)
        gl_1.addWidget(self.le_xmax, 4, 1, 1, 1)
        gl_1.addWidget(self.lbl_ymin, 5, 0, 1, 1)
        gl_1.addWidget(self.le_ymin, 5, 1, 1, 1)
        gl_1.addWidget(self.lbl_ymax, 6, 0, 1, 1)
        gl_1.addWidget(self.le_ymax, 6, 1, 1, 1)
        gl_1.addWidget(self.lbl_mapsheet, 7, 0, 1, 1)
        gl_1.addWidget(self.le_mapsheet, 7, 1, 1, 1)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        gl_1.addWidget(buttonbox, 9, 0, 1, 2)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_sfile.pressed.connect(self.get_sfile)
        self.cmb_bounds.currentIndexChanged.connect(self.change_bounds)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        bounds = None
        ext = ''
        self.change_bounds()

        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return tmp

        if not self.ifile:
            self.showlog('No vector file specified.')
            return False

        txt = self.cmb_bounds.currentText()

        if txt == 'Manual':
            try:
                xmin = float(self.le_xmin.text())
                xmax = float(self.le_xmax.text())
                ymin = float(self.le_ymin.text())
                ymax = float(self.le_ymax.text())
            except ValueError:
                self.showlog('Invalid value in bounds.')
                return False
            bounds = (xmin, ymin, xmax, ymax)
        elif txt == 'SA Mapsheet':
            bounds = maptobounds(self.le_mapsheet.text(), self.crs,
                                 self.showlog)
            if bounds is None:
                return False

        os.chdir(os.path.dirname(self.ifile))

        if 'KML' in ext or '.kml' in self.ifile or '.kmz' in self.ifile:
            gdf = gpd.read_file(self.ifile,  bbox=bounds,
                                allow_unsupported_drivers=True)
        else:
            gdf = gpd.read_file(self.ifile, bbox=bounds, engine='pyogrio')

        if bounds is not None:
            gdf = gdf.clip(mask=bounds)

        gdf = gdf[gdf.geometry != None]
        gdf = gdf.explode(ignore_index=True)

        if gdf.size == 0:
            self.showlog('Unable to load data. Check file or bounds.')
            return False

        if gdf.geom_type.loc[0] == 'Point':
            if 'line' not in gdf.columns:
                gdf['line'] = 'None'
            else:
                gdf['line'] = gdf['line'].astype(str)

        gdf.attrs['source'] = os.path.basename(self.ifile)
        self.outdata['Vector'] = [gdf]

        return True

    def change_bounds(self):
        """Change the bounds combo."""
        txt = self.cmb_bounds.currentText()

        if txt == 'None':
            self.le_xmin.hide()
            self.le_xmax.hide()
            self.le_ymin.hide()
            self.le_ymax.hide()
            self.le_mapsheet.hide()
            self.lbl_xmin.hide()
            self.lbl_xmax.hide()
            self.lbl_ymin.hide()
            self.lbl_ymax.hide()
            self.lbl_mapsheet.hide()
        elif txt == 'Manual':
            self.le_xmin.show()
            self.le_xmax.show()
            self.le_ymin.show()
            self.le_ymax.show()
            self.le_mapsheet.hide()
            self.lbl_xmin.show()
            self.lbl_xmax.show()
            self.lbl_ymin.show()
            self.lbl_ymax.show()
            self.lbl_mapsheet.hide()
        elif txt == 'SA Mapsheet':
            self.le_xmin.hide()
            self.le_xmax.hide()
            self.le_ymin.hide()
            self.le_ymax.hide()
            self.le_mapsheet.show()
            self.lbl_xmin.hide()
            self.lbl_xmax.hide()
            self.lbl_ymin.hide()
            self.lbl_ymax.hide()
            self.lbl_mapsheet.show()

    def get_sfile(self):
        """Get the filename and crs and bounds."""
        self.le_sfile.setText('')

        ext = ('Shapefile (*.shp);;'
               'Zipped Shapefile (*.shp.zip);;'
               'GeoPackage (*.gpkg);;'
               'KML (*.kml);;'
               'KMZ (*.kmz)')

        self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)

        if not self.ifile:
            return False

        self.le_sfile.setText(self.ifile)

        with fiona.open(self.ifile, allow_unsupported_drivers=True) as fio:
            self.crs = fio.crs
            xmin, ymin, xmax, ymax = fio.bounds

        self.le_xmin.setText(str(xmin))
        self.le_xmax.setText(str(xmax))
        self.le_ymin.setText(str(ymin))
        self.le_ymax.setText(str(ymax))

        return True

    def set_bounds(self, bounds):
        """
        Set the bounds.

        Parameters
        ----------
        bounds : list or numpy array
            Bounds defined as (xmin, ymin, xmax, ymax).

        Returns
        -------
        None.

        """
        self.cmb_bounds.setCurrentText('Manual')

        xmin, ymin, xmax, ymax = bounds

        self.le_xmin.setText(str(xmin))
        self.le_xmax.setText(str(xmax))
        self.le_ymin.setText(str(ymin))
        self.le_ymax.setText(str(ymax))

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)


def get_GXYZ(ifile, showlog=print, piter=iter):
    """
    Get Geosoft XYZ.

    Returns
    -------
    df2 : DataFrame
        Pandas dataframe.

    """
    with open(ifile, encoding='utf-8') as fno:
        tmp = fno.read()

    chktxt = tmp[:tmp.index('\n')].lower()

    if r'/' not in chktxt and 'line' not in chktxt and 'tie' not in chktxt:
        showlog('Not Geosoft XYZ format')
        return None

    while r'//' in tmp[:tmp.index('\n')]:
        tmp = tmp[tmp.index('\n')+1:]

    head = None

    while r'/' in tmp[:tmp.index('\n')]:
        head = tmp[:tmp.index('\n')]
        tmp = tmp[tmp.index('\n')+1:]
        head = head.split()
        head.pop(0)

    while r'/' in tmp:
        t1 = tmp[:tmp.index(r'/')]
        t2 = tmp[tmp.index(r'/')+1:]
        t3 = t2[t2.index('\n')+1:]
        tmp = t1+t3

    tmp = tmp.lower()
    tmp = tmp.lstrip()
    tmp = re.split('(line|tie)', tmp)
    if tmp[0] == '':
        tmp.pop(0)

    df2 = None
    dflist = []
    for i in piter(range(0, len(tmp), 2)):
        tmp2 = tmp[i+1]

        line = tmp[i]+' '+tmp2[:tmp2.index('\n')].strip()
        tmp2 = tmp2[tmp2.index('\n')+1:]
        if head is None:
            head = [f'Column {i+1}' for i in
                    range(len(tmp2[:tmp2.index('\n')].split()))]

        tmp2 = tmp2.replace('*', 'NaN')
        df1 = pd.read_csv(StringIO(tmp2), sep=r'\s+', names=head)

        df1['line'] = line
        dflist.append(df1)

    # Concat in all df in one go is much faster
    df2 = pd.concat(dflist, ignore_index=True)

    return df2


def get_intrepid(ifile, showlog=print, piter=iter):
    """
    Get Intrepid Database.

    Returns
    -------
    df : DataFrame
        Pandas Dataframe.

    """
    if '..dir' not in ifile.lower():
        return None

    idir = ifile[:-5]

    dconv = {'IEEE4ByteReal': 'f',
             'IEEE8ByteReal': 'd',
             'Signed32BitInteger': 'i',
             'Signed16BitInteger': 'h'}

    files = glob.glob(os.path.join(idir, '*.PD'))
    vfiles = glob.glob(os.path.join(idir, '*.vec'))

    data = {}
    numbands = {}
    nodata = {}

    for j in piter(range(len(files))):
        ifile = files[j]
        vfile = vfiles[j]
        cname = os.path.basename(ifile)[:-3].lower()

        with open(vfile, encoding='utf-8') as file:
            header = file.readlines()

        for i in header:
            tmp = i.replace('\t', '')
            tmp = tmp.replace('\n', '')
            tmp = tmp.replace(' ', '')

            if 'CellType' in tmp:
                tmp = tmp.split('=')
                celltype = tmp[-1]
            if 'NullCellValue' in tmp:
                tmp = tmp.split('=')
                null = tmp[-1]
            if 'NrOfBands' in tmp:
                tmp = tmp.split('=')
                numbands[cname] = int(tmp[-1])

        fmt = dconv[celltype]
        if fmt in ['i', 'h']:
            null = int(null)
        else:
            null = float(null)
        nodata[cname] = null

        tmp = np.fromfile(ifile, offset=512, dtype=fmt)
        if tmp.min() == -np.inf:
            nodata[cname] = -np.inf

        if numbands[cname] > 1:
            tmp.shape = (tmp.size//numbands[cname], numbands[cname])

        data[cname] = tmp

    if 'linenumber' in data:
        linename = 'linenumber'
        nodata['line'] = nodata[linename]
    else:
        linename = 'line'

    line = data.pop(linename, None)
    data.pop('linetype', None)
    indx = data.pop('index', None)

    i = list(data.keys())[0]
    tmp = data[i].shape
    linenumber = np.zeros(tmp, dtype=int) + nodata[linename]

    for i, indxi in enumerate(indx):
        t1 = indxi[0]
        t2 = t1+indxi[1]+1
        linenumber[t1:t2] = line[i]

    data['line'] = linenumber
    numbands['line'] = 1

    dkeys = list(data.keys())
    for cname in dkeys:
        if numbands[cname] > 1:
            for j in range(numbands[cname]):
                txt = f'{cname}_{j+1}'
                data[txt] = data[cname][:, j]
                nodata[txt] = nodata[cname]
            del data[cname]

    df = pd.DataFrame.from_dict(data)

    df = df.astype(float)
    for col in piter(df.columns):
        df[col].replace(nodata[col], np.nan, inplace=True)

    return df


def _test():
    """Test."""
    import sys
    # from pygmi.misc import ProgressBarText

    # piter = ProgressBarText().iter
    # ifile = r"D:\Additional Survey Data\MAG_MERGE..DIR"
    # ifile = r"D:\Additional Survey Data\RADALL..DIR"

    # data = get_intrepid(ifile, print, piter)

    # ifile = r"E:\WorkProjects\ST-2020-1339 Landslides\vector\landslide polygons_10_sites.kmz"
    ifile = r"D:/Work/Programming/geochem/all_geochem.shp"
    ifile = r"E:\CGS-SpecLib\doc.kml"

    app = QtWidgets.QApplication(sys.argv)

    os.chdir(os.path.dirname(ifile))

    tmp1 = ImportVector()
    # tmp1.idir = r"D:\Landsat"
    # tmp1.idir = r'E:\WorkProjects\ST-2020-1339 Landslides\change'
    # tmp1.get_sfile(True)
    tmp1.settings()

    dat = tmp1.outdata['Vector'][0]

    tmp2 = ColumnSelect()
    tmp2.indata = tmp1.outdata
    tmp2.settings()

    breakpoint()


if __name__ == "__main__":
    _test()
