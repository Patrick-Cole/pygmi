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
""" Import Data """

# pylint: disable=E1101
from PySide import QtGui
import numpy as np
import os
import pygmi.seis.datatypes as sdt


def str2float(inp):
    """ Converts a set number of columns to float, or returns None """
    if inp.strip() == '':
        return None
    return float(inp)


def str2int(inp):
    """ Converts a set number of columns to float, or returns None """
    if inp.strip() == '':
        return None
    return int(inp)


class ImportSeisan(object):
    """ Import Seisan Data """
    def __init__(self, parent=None):
        self.ifile = ""
        self.name = "Import Seisan Data"
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings """
        ext = \
            "Seisan Format (*.out);;" +\
            "All Files (*.*)"

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)[0]
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

        pntfile = open(filename)
        ltmp = pntfile.readlines()
        pntfile.close()

        event = {}
        dat = []

        for i in ltmp:
            if i.strip() == '':
                if len(event) > 0:
                    dat.append(event)
                    event = {}
                continue

            ltype = i[79]
            if ltype == '1' and event.get('1') is None:
                tmp = sdt.seisan_1()
                tmp.year = str2int(i[1:5])
                tmp.month = str2int(i[6:8])
                tmp.day = str2int(i[8:10])
                tmp.fixed_origin_time = i[10]
                tmp.hour = str2int(i[11:13])
                tmp.minutes = str2int(i[13:15])
                tmp.seconds = str2float(i[16:20])
                tmp.location_model_indicator = i[20]
                tmp.distance_indicator = i[21]
                tmp.event_id = i[22]
                tmp.latitude = str2float(i[23:30])
                tmp.longitude = str2float(i[30:38])
                tmp.depth = str2float(i[38:43])
                tmp.depth_indicator = i[43]
                tmp.locating_indicator = i[44]
                tmp.hypocenter_reporting_agency = i[45:48]
                tmp.number_of_stations_used = str2int(i[48:51])
                tmp.rms_of_time_residuals = str2float(i[51:55])
                tmp.magnitude_1 = str2float(i[55:59])
                tmp.type_of_magnitude_1 = i[59]
                tmp.magnitude_reporting_agency_1 = i[60:63]
                tmp.magnitude_2 = str2float(i[63:67])
                tmp.type_of_magnitude_2 = i[67]
                tmp.magnitude_reporting_agency_2 = i[68:71]
                tmp.magnitude_3 = str2float(i[71:75])
                tmp.type_of_magnitude_3 = i[75]
                tmp.magnitude_reporting_agency_3 = i[76:79]
                event['1'] = tmp

            if ltype == 'F':
                tmp = sdt.seisan_F()
                if event.get('F') is None:
                    event['F'] = {}

                tmp.program_used = i[70:77]
                prg = tmp.program_used.strip()
                tmp.strike = str2float(i[0:10])
                tmp.dip = str2float(i[10:20])
                tmp.rake = str2float(i[20:30])
                if prg == 'FPFIT' or prg == 'HASH':
                    tmp.err1 = str2float(i[30:35])
                    tmp.err2 = str2float(i[35:40])
                    tmp.err3 = str2float(i[40:45])
                    tmp.fit_error = str2float(i[45:50])
                    tmp.station_distribution_ratio = str2float(i[50:55])
                if prg == 'FOCMEC' or prg == 'HASH':
                    tmp.amplitude_ratio = str2float(i[55:60])
                if prg == 'FOCMEC' or prg == 'PINV':
                    tmp.number_of_bad_polarities = str2int(i[60:62])
                if prg == 'FOCMEC':
                    tmp.number_of_bad_amplitude_ratios = str2int(i[63:65])
                tmp.agency_code = i[66:69]
                tmp.solution_quality = i[77]
                event['F'][prg] = tmp

            if ltype == 'E':
                tmp = sdt.seisan_E()
                tmp.gap = str2int(i[5:8])
                tmp.origin_time_error = str2float(i[14:20])
                tmp.latitude_error = str2float(i[24:30])
                tmp.longitude_error = str2float(i[32:38])
                tmp.depth_error = str2float(i[38:43])
                tmp.cov_xy = str2float(i[43:55])
                tmp.cov_xz = str2float(i[55:67])
                tmp.cov_yz = str2float(i[67:79])
                event['E'] = tmp

            if ltype == 'I':
                tmp = sdt.seisan_I()
                tmp.last_action_done = i[8:11]
                tmp.date_time_of_last_action = i[12:26]
                tmp.operator = i[30:35]
                tmp.status = i[42:57]
                tmp.id = i[60:74]
                tmp.new_id_created = i[74]
                tmp.id_locked = i[75]
                event['I'] = tmp

            if ltype == '7':
                event['7'] = []
            elif event.get('7') is not None:
                tmp = sdt.seisan_7()

                tmp.stat = i[1:5]
                tmp.sp = i[6:8]
                tmp.iphas = i[9:14]
                tmp.phase_weight = i[14]
                tmp.d = i[16]
                tmp.hour = str2int(i[18:20])
                tmp.minutes = str2int(i[20:22])
                tmp.seconds = str2float(i[23:28])
                tmp.coda = i[29:33]
                tmp.amplitude = str2float(i[34:40])
                tmp.period = str2float(i[41:45])
                tmp.azimuth = str2float(i[46:51])
                tmp.velocity = str2float(i[52:56])
                tmp.angle_incidence = str2float(i[57:60])
                tmp.azimuth_residual = str2int(i[61:63])
                tmp.time_residual = str2float(i[64:68])
                tmp.location_weight = str2int(i[69])
                tmp.distance = str2float(i[72:75])
                tmp.caz = str2int(i[76:79])
                event['7'].append(tmp)

        self.outdata['Seis'] = dat
        return True


class ImportGenericFPS(object):
    """ Import Data """
    def __init__(self, parent=None):
        self.ifile = ""
        self.name = "Import Generic FPS: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings """
        ext = \
            "Comma Delimeted Text (*.csv);;" +\
            "All Files (*.*)"

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)[0]
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

        dlim = ','

        pntfile = open(filename)
        ltmp = pntfile.readline()
        pntfile.close()

        isheader = any(c.isalpha() for c in ltmp)

        srows = 0
        ltmp = ltmp.split(dlim)
        if isheader:
            srows = 1
        else:
            ltmp = [str(c) for c in range(len(ltmp))]

        try:
            datatmp = np.loadtxt(filename, delimiter=dlim, skiprows=srows)
        except ValueError:
            QtGui.QMessageBox.critical(self.parent, 'Import Error',
                                       'There was a problem loading the file.'
                                       ' You may have a text character in one'
                                       ' of your columns.')
            return False

        dat = []
        for i in datatmp:
            event = {}
            tmp = sdt.seisan_1()
            tmp.longitude = i[0]
            tmp.latitude = i[1]
            tmp.magnitude_1 = i[-1]
            event['1'] = tmp

            tmp = sdt.seisan_F()
            tmp.program_used = 'Generic'
            tmp.strike = i[2]
            tmp.dip = i[3]
            tmp.rake = i[4]
            event['F'] = {}
            event['F']['Generic'] = tmp
            dat.append(event)

        self.outdata['Seis'] = dat

        return True

    def gdal(self):
        """ Process """
        dat = []
        bname = self.ifile.split('/')[-1].rpartition('.')[0]+': '
        ifile = self.ifile[:]
        if self.ext == 'hdr':
            ifile = self.ifile[:-4]

        dataset = gdal.Open(ifile, gdal.GA_ReadOnly)
        gtr = dataset.GetGeoTransform()

        for i in range(dataset.RasterCount):
            rtmp = dataset.GetRasterBand(i+1)
            dat.append(dt.Data())
            dat[i].data = np.ma.array(rtmp.ReadAsArray())
            dat[i].data[dat[i].data == rtmp.GetNoDataValue()] = np.nan
            dat[i].data = np.ma.masked_invalid(dat[i].data)

            dat[i].nrofbands = dataset.RasterCount
            dat[i].tlx = gtr[0]
            dat[i].tly = gtr[3]
            dat[i].bandid = bname+str(i+1)
            dat[i].nullvalue = rtmp.GetNoDataValue()
            dat[i].rows = dataset.RasterYSize
            dat[i].cols = dataset.RasterXSize
            dat[i].xdim = abs(gtr[1])
            dat[i].ydim = abs(gtr[5])
            dat[i].gtr = gtr
            dat[i].wkt = dataset.GetProjection()

        self.outdata['Raster'] = dat

class ExportSeisan(object):
    """ Export Data """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Export Point: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.lmod = None
#        self.dirname = ""
        self.showtext = self.parent.showprocesslog

    def run(self):
        """ Show Info """
        if 'Point' not in self.indata:
            self.parent.showprocesslog(
                'Error: You need to have a point data first!')
            return

        data = self.indata['Point']

        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'csv (*.csv)')[0]

        if filename == '':
            return

        os.chdir(filename.rpartition('/')[0])
        ofile = str(filename.rpartition('/')[-1][:-4])
        self.ext = filename[-3:]

        for i in range(len(data)):
            datid = data[i].dataid
            if datid is '':
                datid = str(i)

            dattmp = np.transpose([data[i].xdata, data[i].ydata,
                                  data[i].zdata])

            ofile2 = ofile+'_'+''.join(x for x in datid if x.isalnum())+'.csv'

            np.savetxt(ofile2, dattmp, delimiter=',')
