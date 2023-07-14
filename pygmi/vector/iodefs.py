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

from pygmi import menu_default
from pygmi.raster.dataprep import GroupProj
from pygmi.misc import BasicModule, ContextModule


class ImportXYZ(BasicModule):
    """
    Import XYZ Data.

    This class imports tabular data.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filt = ''

        self.xchan = QtWidgets.QComboBox()
        self.ychan = QtWidgets.QComboBox()
        self.nodata = QtWidgets.QLineEdit('99999')
        self.proj = GroupProj('Input Projection')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.vector.iodefs.'
                                           'importxyzdata')
        label_xchan = QtWidgets.QLabel('X Channel:')
        label_ychan = QtWidgets.QLabel('Y Channel:')
        label_nodata = QtWidgets.QLabel('Nodata Value:')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.xchan.setSizeAdjustPolicy(3)
        self.ychan.setSizeAdjustPolicy(3)
        self.xchan.setMinimumSize = 10
        self.ychan.setMinimumSize = 10

        self.setWindowTitle(r'Import XYZ Data')

        gridlayout_main.addWidget(label_xchan, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.xchan, 0, 1, 1, 1)

        gridlayout_main.addWidget(label_ychan, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.ychan, 1, 1, 1, 1)

        gridlayout_main.addWidget(label_nodata, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.nodata, 2, 1, 1, 1)

        gridlayout_main.addWidget(helpdocs, 5, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 5, 1, 1, 3)

        gridlayout_main.addWidget(self.proj, 3, 0, 1, 4)

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
            ext = ('Excel (*.xlsx);;'
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

        elif self.filt == 'Excel (*.xlsx)':
            gdf = self.get_excel()
        elif self.filt == 'ASCII XYZ (*.xyz)':
            gdf = self.get_delimited(' ')
        elif self.filt == 'Comma Delimited (*.csv)':
            gdf = self.get_delimited(',')
        elif self.filt == 'Tab Delimited (*.txt)':
            gdf = self.get_delimited('\t')
        elif self.filt == 'Space Delimited (*.txt)':
            gdf = self.get_delimited(' ')
        elif self.filt == 'Intrepid Database (*..DIR)':
            gdf = get_intrepid(self.ifile, self.showlog, self.piter)
            self.nodata.setDisabled(True)
        else:
            return False

        if gdf is None:
            return False

        self.proj.set_current('None')

        self.xchan.clear()
        self.ychan.clear()

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
                    self.proj.combodatum.setCurrentIndex(1)
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

        self.xchan.addItems(gdf.columns.values)
        self.ychan.addItems(gdf.columns.values)

        self.xchan.setCurrentIndex(xind)
        self.ychan.setCurrentIndex(yind)

        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return tmp

        try:
            nodata = float(self.nodata.text())
        except ValueError:
            self.showlog('Nodata Value error - abandoning import')
            return False

        xcol = self.xchan.currentText()
        ycol = self.ychan.currentText()

        x = gdf[xcol]
        y = gdf[ycol]

        if x.dtype == 'O' or y.dtype == 'O':
            self.showlog('Error: You have text in your coordinates.')
            return False

        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(x, y))

        gdf['line'] = gdf['line'].astype(str)

        if self.nodata.isEnabled():
            gdf = gdf.replace(nodata, np.nan)

        if self.proj.wkt != '':
            gdf = gdf.set_crs(self.proj.wkt)

        gdf.attrs['source'] = os.path.basename(self.ifile)
        self.outdata['Vector'] = [gdf]

        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        self.ifile = projdata['ifile']
        self.filt = projdata['filt']
        self.xchan.setCurrentText(projdata['xchan'])
        self.ychan.setCurrentText(projdata['ychan'])
        self.nodata.setText(projdata['nodata'])

        chk = self.settings(True)

        return chk

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        projdata['ifile'] = self.ifile
        projdata['filt'] = self.filt
        projdata['xchan'] = self.xchan.currentText()
        projdata['ychan'] = self.ychan.currentText()
        projdata['nodata'] = self.nodata.text()

        return projdata

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

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'csv (*.csv);; Excel (*.xlsx)')

        if filename == '':
            self.parent.process_is_active(False)
            return False

        self.showlog('Export busy...')

        os.chdir(os.path.dirname(filename))
        data = self.indata['Vector'][0]

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
            ext = 'Shapefile (*.shp);;Zipped Shapefile (*.shp.zip);;GeoPackage (*.gpkg)'

            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent,
                                                                  'Open File',
                                                                  '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))

        gdf = gpd.read_file(self.ifile, engine='pyogrio')
        gdf = gdf[gdf.geometry != None]
        gdf = gdf.explode(ignore_index=True)

        if gdf.geom_type.loc[0] == 'Point':
            if 'line' not in gdf.columns:
                gdf['line'] = 'None'
            else:
                gdf['line'] = gdf['line'].astype(str)

        gdf.attrs['source'] = os.path.basename(self.ifile)
        self.outdata['Vector'] = [gdf]

        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        self.ifile = projdata['ifile']

        chk = self.settings(True)

        return chk

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        projdata['ifile'] = self.ifile

        return projdata


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
        df1 = pd.read_csv(StringIO(tmp2), sep='\s+', names=head)

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
        # print(cname)

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

    for i, _ in enumerate(indx):
        t1 = indx[i][0]
        t2 = t1+indx[i][1]+1
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
    from pygmi.misc import ProgressBarText

    piter = ProgressBarText().iter
    ifile = r"D:\Additional Survey Data\MAG_MERGE..DIR"
    ifile = r"D:\Additional Survey Data\RADALL..DIR"

    data = get_intrepid(ifile, print, piter)

    breakpoint()


if __name__ == "__main__":
    _test()
