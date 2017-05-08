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

import os
import re
from PyQt5 import QtWidgets
import numpy as np
import pygmi.seis.datatypes as sdt


def str2float(inp):
    """
    Converts a set number of columns to float, or returns None

    Parameters
    ----------
    inp : str
        string with a list of floats in it

    Returns
    -------
    output : float
        all columns returned as floats
    """
    if inp.strip() == '':
        return None
    return float(inp)


def str2int(inp):
    """
    Converts a set number of columns to integer, or returns None

    Parameters
    ----------
    inp : str
        string with a list of floats in it

    Returns
    -------
    output : float
        all columns returned as integers
    """
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

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

        pntfile = open(filename)
        ltmp = pntfile.readlines()
        pntfile.close()

        if len(ltmp[0]) < 80:
            self.parent.showprocesslog('Error: Problem with file')
            return False

        # This constructs a dictionary of functions
        read_record_type = {}
        read_record_type['1'] = read_record_type_1
        read_record_type['2'] = read_record_type_2
        read_record_type['4'] = read_record_type_4
        read_record_type['5'] = read_record_type_5
        read_record_type['6'] = read_record_type_6
        read_record_type['E'] = read_record_type_e
        read_record_type['F'] = read_record_type_f
        read_record_type['H'] = read_record_type_h
        read_record_type['I'] = read_record_type_i
        read_record_type['M'] = read_record_type_m
        read_record_type['P'] = read_record_type_p

        event = {}
        event['4'] = []
        event['F'] = {}
        dat = []

        for i in ltmp:
            if i.strip() == '':
                if event:
                    dat.append(event)
                    event = {}
                    event['4'] = []
                    event['F'] = {}
                continue

            ltype = i[79]

            if ltype == '1' and event.get('1') is not None:
                continue

            if ltype == '7' or ltype == '3':
                continue

            if ltype == ' ':
                ltype = '4'

            if ltype == 'F':
                event[ltype].update(read_record_type[ltype](i))
            elif ltype == '4':
                event[ltype].append(read_record_type[ltype](i))
            elif ltype == 'M' and event.get('M') is not None:
                event[ltype] = read_record_type[ltype](i, event[ltype])
            else:
                try:
                    event[ltype] = read_record_type[ltype](i)
                except:
                    self.parent.showprocesslog('Error: Problem with file on line:')
                    self.parent.showprocesslog(i)
                    return False

        if event:
            dat.append(event)

        self.outdata['Seis'] = dat
        return True


def read_record_type_1(i):
    """ Reads record type 1"""

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

    return tmp


def read_record_type_2(i):
    """ Reads record type 2"""

    dat = sdt.seisan_2()
    dat.description = i[5:20]
    dat.diastrophism_code = i[21]
    dat.tsunami_code = i[22]
    dat.seiche_code = i[23]
    dat.cultural_effects = i[24]
    dat.unusual_events = i[25]
    dat.max_intensity = str2int(i[27:29])
    dat.max_intensity_qualifier = i[29]
    dat.intensity_scale = i[30:32]
    dat.macroseismic_latitude = str2float(i[33:39])
    dat.macroseismic_longitude = str2float(i[40:47])
    dat.macroseismic_magnitude = str2float(i[48:51])
    dat.type_of_magnitude = i[51]
    dat.log_of_felt_area_radius = str2float(i[52:56])
    dat.log_of_area_1 = str2float(i[56:61])
    dat.intensity_bordering_area_1 = str2int(i[61:63])
    dat.log_of_area_2 = str2float(i[63:68])
    dat.intensity_bordering_area_2 = str2int(i[68:70])
    dat.quality_rank = i[71]
    dat.reporting_agency = i[72:75]

    return dat


def read_record_type_4(i):
    """ Reads record type 4"""

    tmp = sdt.seisan_4()

    tmp.station_name = i[1:6]
    tmp.instrument_type = i[6]
    tmp.component = i[7]
    tmp.quality = i[9]
    tmp.phase_id = i[10:14]
    tmp.weighting_indicator = str2int(i[14])
    tmp.flag_auto_pick = i[15]
    tmp.first_motion = i[16]
    tmp.hour = str2int(i[18:20])
    tmp.minutes = str2int(i[20:22])
    tmp.seconds = str2float(i[22:28])
    tmp.duration = str2int(i[29:33])
    tmp.amplitude = str2float(i[33:40])
    tmp.period = str2float(i[41:45])
    tmp.direction_of_approach = str2float(i[46:51])
    tmp.phase_velocity = str2float(i[52:56])
    tmp.angle_of_incidence = str2float(i[56:60])
    tmp.azimuth_residual = str2int(i[60:63])
    tmp.travel_time_residual = str2float(i[63:68])
    tmp.weight = str2int(i[68:70])
    tmp.epicentral_distance = str2float(i[70:75])
    tmp.azimuth_at_source = str2int(i[76:79])

    return tmp


def read_record_type_5(i):
    """ Reads record type 5"""

    tmp = sdt.seisan_5()
    tmp.text = i[1:79]
    return tmp


def read_record_type_6(i):
    """ Reads record type 6"""

    tmp = sdt.seisan_6()
    tmp.tracedata_files = i[1:79]

    return tmp


def read_record_type_e(i):
    """ Reads record type E"""

    tmp = sdt.seisan_E()

    tmp.gap = str2int(i[5:8])
    tmp.origin_time_error = str2float(i[14:20])
    tmp.latitude_error = str2float(i[24:30])
    tmp.longitude_error = str2float(i[32:38])
    tmp.depth_error = str2float(i[38:43])
    tmp.cov_xy = str2float(i[43:55])
    tmp.cov_xz = str2float(i[55:67])
    tmp.cov_yz = str2float(i[67:79])

    return tmp


def read_record_type_f(i):
    """ Reads record type F"""

    tmp = sdt.seisan_F()

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
    tmp.agency_code = i[66:69]
    tmp.solution_quality = i[77]

    if prg == '':
        out = {}
    else:
        out = {prg: tmp}

    return out


def read_record_type_h(i):
    """ Reads record type H"""

    tmp = sdt.seisan_H()
    tmp.year = str2int(i[1:5])
    tmp.month = str2int(i[6:8])
    tmp.day = str2int(i[8:10])
    tmp.fixed_origin_time = i[10]
    tmp.hour = str2int(i[11:13])
    tmp.minutes = str2int(i[13:15])
    tmp.seconds = str2float(i[16:22])
    tmp.latitude = str2float(i[23:32])
    tmp.longitude = str2float(i[33:43])
    tmp.depth = str2float(i[44:52])
    tmp.rms = str2float(i[53:59])

    return tmp


def read_record_type_i(i):
    """ Reads record type I"""

    tmp = sdt.seisan_I()

    tmp.last_action_done = i[8:11]
    tmp.date_time_of_last_action = i[12:26]
    tmp.operator = i[30:35]
    tmp.status = i[42:57]
    tmp.id = i[60:74]
    tmp.new_id_created = i[74]
    tmp.id_locked = i[75]

    return tmp


def read_record_type_m(i, vtmp=None):
    """ Reads record type M"""

    if i[1:3] is not 'MT':
        tmp = sdt.seisan_M()
        tmp.year = i[1:5]
        tmp.month = i[6:8]
        tmp.day = i[8:10]
        tmp.hour = i[11:13]
        tmp.minutes = i[13:15]
        tmp.seconds = i[16:20]
        tmp.latitude = i[23:30]
        tmp.longitude = i[30:38]
        tmp.depth = i[38:43]
        tmp.reporting_agency = i[45:48]
        tmp.magnitude = i[55:59]
        tmp.magnitude_type = i[59]
        tmp.magnitude_reporting_agency = i[60:63]
        tmp.method_used = i[70:77]
        tmp.quality = i[77]
    else:
        tmp = vtmp['M']
        tmp.mrr_mzz = i[3:9]
        tmp.mtt_mxx = i[10:16]
        tmp.mpp_myy = i[17:23]
        tmp.mrt_mzx = i[24:30]
        tmp.mrp_mzy = i[31:37]
        tmp.mtp_mxy = i[38:44]
        tmp.reporting_agency2 = i[45:48]
        tmp.mt_coordinate_system = i[48]
        tmp.exponential = i[49:51]
        tmp.scalar_moment = i[52:62]
        tmp.method_used_2 = i[70:77]
        tmp.quality_2 = i[77]

    return tmp


def read_record_type_p(i):
    """ Reads record type P"""

    tmp = sdt.seisan_P()
    tmp.filename = i[1:79]
    return tmp


class ImportGenericFPS(object):
    """
    Import Generic Fault Plane Solution Data. This is stored in a csv file.
    """
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
        QtWidgets.QMessageBox.information(self.parent, 'File Format',
                                          'The file should have the following '
                                          'columns: longitude, latitude, '
                                          'depth, strike, dip, rake, '
                                          'magnitude.')

        ext = \
            "Comma Delimeted Text (*.csv);;" +\
            "All Files (*.*)"

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
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
            QtWidgets.QMessageBox.critical(self.parent, 'Import Error',
                                           'There was a problem loading the '
                                           'file. You may have a text '
                                           'character in one of your columns.')
            return False

        dat = []
        if datatmp.ndim == 1:
            datatmp = np.expand_dims(datatmp, 0)

        for i in datatmp:
            event = {}
            tmp = sdt.seisan_1()
            tmp.longitude = i[0]
            tmp.latitude = i[1]
            tmp.depth = i[2]
            tmp.magnitude_1 = i[-1]
            event['1'] = tmp

            tmp2 = sdt.seisan_F()
            tmp2.program_used = 'Generic'
            tmp2.strike = i[3]
            tmp2.dip = i[4]
            tmp2.rake = i[5]
            event['F'] = {}
            event['F']['Generic'] = tmp2
            dat.append(event)

        self.outdata['Seis'] = dat

        return True


class ExportSeisan(object):
    """ Export Seisan Data """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Export Seisan Data "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.lmod = None
        self.fobj = None
        self.showtext = self.parent.showprocesslog

    def run(self):
        """ Show Info """
        if 'Seis' not in self.indata:
            self.parent.showprocesslog(
                'Error: You need to have a Seisan data first!')
            return

        data = self.indata['Seis']

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self.parent,
                                                            'Save File',
                                                            '.', 'sei (*.sei)')

        if filename == '':
            return

        os.chdir(filename.rpartition('/')[0])
        self.ext = filename[-3:]

        self.fobj = open(filename, 'w')

        for i in data:
            self.write_record_type_1(i)
            self.write_record_type_f(i)  # This is missing  some #3 recs
            self.write_record_type_m(i)  # This is missing  some #3 recs
            self.write_record_type_e(i)
            self.write_record_type_i(i)  # This is missing  some #3 recs
            self.write_record_type_4(i)
            self.fobj.write(' \n')

        self.fobj.close()

    def write_record_type_1(self, data):
        """ Writes record type 1"""
        if '1' not in data:
            return
        else:
            dat = data['1']
        tmp = ' '*80+'\n'
        tmp = sform('{0:4d}', dat.year, tmp, 2, 5)
        tmp = sform('{0:2d}', dat.month, tmp, 7, 8)
        tmp = sform('{0:2d}', dat.day, tmp, 9, 10)
        tmp = sform('{0:1s}', dat.fixed_origin_time, tmp, 11)
        tmp = sform('{0:2d}', dat.hour, tmp, 12, 13)
        tmp = sform('{0:2d}', dat.minutes, tmp, 14, 15)
        tmp = sform('{0:4.1f}', dat.seconds, tmp, 17, 20)
        tmp = sform('{0:1s}', dat.location_model_indicator, tmp, 21)
        tmp = sform('{0:1s}', dat.distance_indicator, tmp, 22)
        tmp = sform('{0:1s}', dat.event_id, tmp, 23)
        tmp = sform('{0:7.3f}', dat.latitude, tmp, 24, 30)
        tmp = sform('{0:8.3f}', dat.longitude, tmp, 31, 38)
        tmp = sform('{0:5.1f}', dat.depth, tmp, 39, 43)
        tmp = sform('{0:1s}', dat.depth_indicator, tmp, 44)
        tmp = sform('{0:1s}', dat.locating_indicator, tmp, 45)
        tmp = sform('{0:3s}', dat.hypocenter_reporting_agency, tmp, 46, 48)
        tmp = sform('{0:3d}', dat.number_of_stations_used, tmp, 49, 51)
        tmp = sform('{0:4.1f}', dat.rms_of_time_residuals, tmp, 52, 55)
        tmp = sform('{0:4.1f}', dat.magnitude_1, tmp, 56, 59)
        tmp = sform('{0:1s}', dat.type_of_magnitude_1, tmp, 60)
        tmp = sform('{0:3s}', dat.magnitude_reporting_agency_1, tmp, 61, 63)
        tmp = sform('{0:4.1f}', dat.magnitude_2, tmp, 64, 67)
        tmp = sform('{0:1s}', dat.type_of_magnitude_2, tmp, 68)
        tmp = sform('{0:3s}', dat.magnitude_reporting_agency_2, tmp, 69, 71)
        tmp = sform('{0:4.1f}', dat.magnitude_3, tmp, 72, 75)
        tmp = sform('{0:1s}', dat.type_of_magnitude_3, tmp, 76, 76)
        tmp = sform('{0:3s}', dat.magnitude_reporting_agency_3, tmp, 77, 79)
        tmp = sform('{0:1s}', '1', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_2(self, data):
        """ Writes record type 2"""
        if '2' not in data:
            return
        else:
            dat = data['2']

        tmp = ' '*80+'\n'
        tmp = sform('{0:15s}', dat.description, tmp, 6, 20)
        tmp = sform('{0:1s}', dat.diastrophism_code, tmp, 22)
        tmp = sform('{0:1s}', dat.tsunami_code, tmp, 23)
        tmp = sform('{0:1s}', dat.seiche_code, tmp, 24)
        tmp = sform('{0:1s}', dat.cultural_effects, tmp, 25)
        tmp = sform('{0:1s}', dat.unusual_events, tmp, 26)
        tmp = sform('{0:2d}', dat.max_intensity, tmp, 28, 29)
        tmp = sform('{0:1s}', dat.max_intensity_qualifier, tmp, 30)
        tmp = sform('{0:2s}', dat.intensity_scale, tmp, 31, 32)
        tmp = sform('{0:6.2f}', dat.macroseismic_latitude, tmp, 34, 39)
        tmp = sform('{0:7.2f}', dat.macroseismic_longitude, tmp, 41, 47)
        tmp = sform('{0:3.1f}', dat.macroseismic_magnitude, tmp, 49, 51)
        tmp = sform('{0:1s}', dat.type_of_magnitude, tmp, 52)
        tmp = sform('{0:4.2f}', dat.log_of_felt_area_radius, tmp, 53, 56)
        tmp = sform('{0:5.2f}', dat.log_of_area_1, tmp, 57, 61)
        tmp = sform('{0:2d}', dat.intensity_bordering_area_1, tmp, 62, 63)
        tmp = sform('{0:5.2f}', dat.log_of_area_2, tmp, 64, 68)
        tmp = sform('{0:2d}', dat.intensity_bordering_area_2, tmp, 69, 70)
        tmp = sform('{0:1s}', dat.quality_rank, tmp, 72)
        tmp = sform('{0:3s}', dat.reporting_agency, tmp, 73, 75)
        tmp = sform('{0:1s}', '2', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_3(self, tmp):
        """ Writes record type 3 - this changes depending on the preceding line
        """
        if '3' not in tmp:
            return

        tmp = ' '*80+'\n'
        tmp = sform('{0:1s}', '3', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_4(self, data):
        """ Writes record type 4"""
        if '4' not in data:
            return
        else:
            dat = data['4']

        self.write_record_type_7()

        for dat in data['4']:
            tmp = ' '*80+'\n'
            tmp = sform('{0:5s}', dat.station_name, tmp, 2, 6)
            tmp = sform('{0:1s}', dat.instrument_type, tmp, 7)
            tmp = sform('{0:1s}', dat.component, tmp, 8)
            tmp = sform('{0:1s}', dat.quality, tmp, 10)
            tmp = sform('{0:5s}', dat.phase_id, tmp, 11, 14)
            tmp = sform('{0:1d}', dat.weighting_indicator, tmp, 15, 15, 1)
            tmp = sform('{0:1s}', dat.flag_auto_pick, tmp, 16)
            tmp = sform('{0:1s}', dat.first_motion, tmp, 17)
            tmp = sform('{0:2d}', dat.hour, tmp, 19, 20, 0)
            tmp = sform('{0:2d}', dat.minutes, tmp, 21, 22, 0)
            tmp = sform('{0:>6.2f}', dat.seconds, tmp, 23, 28, 0)
            tmp = sform('{0:4d}', dat.duration, tmp, 30, 33)
            tmp = sform('{0:>7.5G}', dat.amplitude, tmp, 34, 40, 0)
            tmp = sform('{0:4.3G}', dat.period, tmp, 42, 45, 0)
            tmp = sform('{0:5.1f}', dat.direction_of_approach, tmp, 47, 51)
            tmp = sform('{0:4.0f}', dat.phase_velocity, tmp, 53, 56)
            tmp = sform('{0:4.0f}', dat.angle_of_incidence, tmp, 57, 60)
            tmp = sform('{0:3d}', dat.azimuth_residual, tmp, 61, 63)
            tmp = sform('{0:5.2f}', dat.travel_time_residual, tmp, 64, 68, 0)
            tmp = sform('{0:2d}', dat.weight, tmp, 69, 70)
            tmp = sform('{0:5.4G}', dat.epicentral_distance, tmp, 71, 75, 0)
            tmp = sform('{0:3d}', dat.azimuth_at_source, tmp, 77, 79, 0)

            self.fobj.write(tmp)

    def write_record_type_5(self, data):
        """ Writes record type 5"""
        if '5' not in data:
            return
        else:
            dat = data['5']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.text, tmp, 2, 79)
        tmp = sform('{0:1s}', '5', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_6(self, data):
        """ Writes record type 6"""
        if '6' not in data:
            return
        else:
            dat = data['6']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.tracedata_files, tmp, 2, 79)
        tmp = sform('{0:1s}', '6', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_7(self):
        """ Writes record type 7 """
        tmp = ' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO' + \
            ' AIN AR TRES W  DIS CAZ7\n'
        self.fobj.write(tmp)

    def write_record_type_e(self, data):
        """ Writes record type E"""
        if 'E' not in data:
            return
        else:
            dat = data['E']

        if dat.latitude_error == -999 or dat.latitude_error is None:
            return

        tmp = ' '*80+'\n'
        tmp = sform('{0:4s}', 'GAP=', tmp, 2, 5)
        tmp = sform('{0:3d}', dat.gap, tmp, 6, 8)
        tmp = sform('{0:6.2f}', dat.origin_time_error, tmp, 15, 20)
        tmp = sform('{0:6.1f}', dat.latitude_error, tmp, 25, 30)
        tmp = sform('{0:6.1f}', dat.longitude_error, tmp, 33, 38)
        tmp = sform('{0:5.1f}', dat.depth_error, tmp, 39, 43)
        tmp = sform('{0:12.4E}', dat.cov_xy, tmp, 44, 55)
        tmp = sform('{0:12.4E}', dat.cov_xz, tmp, 56, 67)
        tmp = sform('{0:12.4E}', dat.cov_yz, tmp, 68, 79)
        tmp = sform('{0:1s}', 'E', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_f(self, data):
        """ Writes record type F"""

        if 'F' not in data:
            return

        for dat in data['F'].values():
            tmp = ' '*80+'\n'
            tmp = sform('{0:10.1f}', dat.strike, tmp, 1, 10)
            tmp = sform('{0:10.1f}', dat.dip, tmp, 11, 20)
            tmp = sform('{0:10.1f}', dat.rake, tmp, 21, 30)
            tmp = sform('{0:5.1f}', dat.err1, tmp, 31, 35)
            tmp = sform('{0:5.1f}', dat.err2, tmp, 36, 40)
            tmp = sform('{0:5.1f}', dat.err3, tmp, 41, 45)
            tmp = sform('{0:5.1f}', dat.fit_error, tmp, 46, 50)
            tmp = sform('{0:5.1f}', dat.station_distribution_ratio, tmp, 51,
                        55)
            tmp = sform('{0:5.1f}', dat.amplitude_ratio, tmp, 56, 60)
            tmp = sform('{0:2d}', dat.number_of_bad_polarities, tmp, 61, 62)
            tmp = sform('{0:2d}', dat.number_of_bad_amplitude_ratios, tmp, 64,
                        65)
            tmp = sform('{0:3s}', dat.agency_code, tmp, 67, 69)
            tmp = sform('{0:7s}', dat.program_used, tmp, 71, 77)
            tmp = sform('{0:1s}', dat.solution_quality, tmp, 78)
            tmp = sform('{0:1s}', 'F', tmp, 80)

            self.fobj.write(tmp)

    def write_record_type_h(self, data):
        """ Writes record type H"""

        if 'H' not in data:
            return
        else:
            dat = data['1']

        tmp = ' '*80+'\n'
        tmp = sform('{0:4d}', dat.year, tmp, 2, 5)
        tmp = sform('{0:2d}', dat.month, tmp, 7, 8)
        tmp = sform('{0:2d}', dat.day, tmp, 9, 10)
        tmp = sform('{0:1s}', dat.fixed_origin_time, tmp, 11)
        tmp = sform('{0:2d}', dat.hour, tmp, 12, 13)
        tmp = sform('{0:2d}', dat.minutes, tmp, 14, 15)
        tmp = sform('{0:6.3f}', dat.seconds, tmp, 17, 22)
        tmp = sform('{0:9.5f}', dat.latitude, tmp, 24, 32)
        tmp = sform('{0:10.5f}', dat.longitude, tmp, 34, 43)
        tmp = sform('{0:8.3f}', dat.depth, tmp, 45, 52)
        tmp = sform('{0:6.3f}', dat.depth, tmp, 54, 59)
        tmp = sform('{0:1s}', 'H', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_i(self, data):
        """ Writes record type I"""
        if 'I' not in data:
            return
        else:
            dat = data['I']

        tmp = ' '*80+'\n'
        tmp = sform('{0:7s}', 'ACTION:', tmp, 2, 8)
        tmp = sform('{0:3s}', dat.last_action_done, tmp, 9, 11)
        tmp = sform('{0:14s}', dat.date_time_of_last_action, tmp, 13, 26)
        tmp = sform('{0:3s}', 'OP:', tmp, 28, 30)
        tmp = sform('{0:4s}', dat.operator, tmp, 31, 34)
        tmp = sform('{0:7s}', 'STATUS:', tmp, 36, 42)
        tmp = sform('{0:14s}', dat.status, tmp, 43, 56)
        tmp = sform('{0:3s}', 'ID:', tmp, 58, 60)
        tmp = sform('{0:14s}', dat.id, tmp, 61, 74)
        tmp = sform('{0:1s}', dat.new_id_created, tmp, 75)
        tmp = sform('{0:1s}', dat.id_locked, tmp, 76)
        tmp = sform('{0:1s}', 'I', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_m(self, data):
        """ Writes record type M"""

        if 'M' not in data:
            return
        else:
            dat = data['M']

        tmp = ' '*80+'\n'

        tmp = sform('{0:4d}', dat.year, tmp, 2, 5)
        tmp = sform('{0:2d}', dat.month, tmp, 7, 8)
        tmp = sform('{0:2d}', dat.day, tmp, 9, 10)
        tmp = sform('{0:2d}', dat.hour, tmp, 12, 13)
        tmp = sform('{0:2d}', dat.minutes, tmp, 14, 15)
        tmp = sform('{0:4.1f}', dat.seconds, tmp, 17, 20)
        tmp = sform('{0:7.3f}', dat.latitude, tmp, 24, 30)
        tmp = sform('{0:8.3f}', dat.longitude, tmp, 31, 38)
        tmp = sform('{0:5.1f}', dat.depth, tmp, 39, 43)
        tmp = sform('{0:3s}', dat.reporting_agency, tmp, 46, 48)
        tmp = sform('{0:4.1f}', dat.magnitude, tmp, 56, 59)
        tmp = sform('{0:1s}', dat.magnitude_type, tmp, 60)
        tmp = sform('{0:3s}', dat.magnitude_reporting_agency, tmp, 61, 63)
        tmp = sform('{0:7s}', dat.method_used, tmp, 71, 77)
        tmp = sform('{0:1s}', dat.quality, tmp, 78)

        tmp = sform('{0:1s}', 'M', tmp, 80)

        self.fobj.write(tmp)

        tmp = ' '*80+'\n'

        tmp = sform('{0:2s}', 'MT', tmp, 2, 3)
        tmp = sform('{0:6.3f}', dat.mrr_mzz, tmp, 4, 9)
        tmp = sform('{0:6.3f}', dat.mtt_mxx, tmp, 11, 16)
        tmp = sform('{0:6.3f}', dat.mpp_myy, tmp, 18, 23)
        tmp = sform('{0:6.3f}', dat.mrt_mzx, tmp, 25, 30)
        tmp = sform('{0:6.3f}', dat.mrp_mzy, tmp, 32, 37)
        tmp = sform('{0:6.3f}', dat.mtp_mxy, tmp, 39, 44)
        tmp = sform('{0:3s}', dat.reporting_agency2, tmp, 46, 48)
        tmp = sform('{0:1s}', dat.mt_coordinate_system, tmp, 49)
        tmp = sform('{0:2d}', dat.exponential, tmp, 50, 51)
        tmp = sform('{0:6.3g}', dat.scalar_moment, tmp, 53, 62)
        tmp = sform('{0:7s}', dat.method_used_2, tmp, 71, 77)
        tmp = sform('{0:1s}', dat.quality_2, tmp, 78)
        tmp = sform('{0:1s}', 'M', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_p(self, data):
        """ Writes record type P"""

        if 'P' not in data:
            return
        else:
            dat = data['P']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.filename, tmp, 2, 79)
        tmp = sform('{0:1s}', 'P', tmp, 80)

        self.fobj.write(tmp)


class ExportCSV(object):
    """ Export Seisan Data """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Export CSV Data "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.lmod = None
        self.fobj = None
        self.showtext = self.parent.showprocesslog

    def run(self):
        """ Show Info """
        if 'Seis' not in self.indata:
            self.parent.showprocesslog(
                'Error: You need to have a Seisan data first!')
            return

        data = self.indata['Seis']

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self.parent,
                                                            'Save File',
                                                            '.', 'csv (*.csv)')
        if filename == '':
            return
        os.chdir(filename.rpartition('/')[0])
        self.ext = filename[-3:]

        self.fobj = open(filename, 'w')

        headi = ('last_action_done, date_time_of_last_action, operator, '
                 'status, id, new_id_created, id_locked, ')

        head1 = ('year, month, day, fixed_origin_time, hour, minutes, '
                 'seconds, location_model_indicator, distance_indicator, '
                 'event_id, latitude, longitude, depth, depth_indicator, '
                 'locating_indicator, hypocenter_reporting_agency, '
                 'number_of_stations_used, rms_of_time_residuals, '
                 'magnitude_1, type_of_magnitude_1, '
                 'magnitude_reporting_agency_1, magnitude_2, '
                 'type_of_magnitude_2, magnitude_reporting_agency_2, '
                 'magnitude_3, type_of_magnitude_3, '
                 'magnitude_reporting_agency_3, ')

        heade = ('gap, origin_time_error, latitude_error, longitude_error, '
                 'depth_error, cov_xy, cov_xz, cov_yz, ')

        head4 = ('station_name, instrument_type, dat.component, quality, '
                 'phase_id, weighting_indicator, flag_auto_pick, '
                 'first_motion, hour, minutes, seconds, duration, amplitude, '
                 'period, direction_of_approach, phase_velocity, '
                 'angle_of_incidence, azimuth_residual, travel_time_residual, '
                 'weight, epicentral_distance, azimuth_at_source, ')

        self.fobj.write(headi+head1+heade+head4+'\n')

        for i in data:
            reci = self.write_record_type_i(i)
            rec1 = self.write_record_type_1(i)
            rece = self.write_record_type_e(i)
            rec4 = self.write_record_type_4(i)
            for j in rec4:
                self.fobj.write(reci+rec1+rece+j+'\n')

        self.fobj.close()

    def write_record_type_1(self, data):
        """ Writes record type 1"""
        if '1' not in data:
            return ', '*27
        else:
            dat = data['1']
        tmp = str(dat.year)+', '
        tmp += str(dat.month)+', '
        tmp += str(dat.day)+', '
        tmp += str(dat.fixed_origin_time)+', '
        tmp += str(dat.hour)+', '
        tmp += str(dat.minutes)+', '
        tmp += str(dat.seconds)+', '
        tmp += str(dat.location_model_indicator)+', '
        tmp += str(dat.distance_indicator)+', '
        tmp += str(dat.event_id)+', '
        tmp += str(dat.latitude)+', '
        tmp += str(dat.longitude)+', '
        tmp += str(dat.depth)+', '
        tmp += str(dat.depth_indicator)+', '
        tmp += str(dat.locating_indicator)+', '
        tmp += str(dat.hypocenter_reporting_agency)+', '
        tmp += str(dat.number_of_stations_used)+', '
        tmp += str(dat.rms_of_time_residuals)+', '
        tmp += str(dat.magnitude_1)+', '
        tmp += str(dat.type_of_magnitude_1)+', '
        tmp += str(dat.magnitude_reporting_agency_1)+', '
        tmp += str(dat.magnitude_2)+', '
        tmp += str(dat.type_of_magnitude_2)+', '
        tmp += str(dat.magnitude_reporting_agency_2)+', '
        tmp += str(dat.magnitude_3)+', '
        tmp += str(dat.type_of_magnitude_3)+', '
        tmp += str(dat.magnitude_reporting_agency_3)+', '

        tmp = tmp.replace('None', '')
        return tmp

    def write_record_type_2(self, data):
        """ Writes record type 2"""
        if '2' not in data:
            return
        else:
            dat = data['2']

        tmp = ' '*80+'\n'
        tmp = sform('{0:15s}', dat.description, tmp, 6, 20)
        tmp = sform('{0:1s}', dat.diastrophism_code, tmp, 22)
        tmp = sform('{0:1s}', dat.tsunami_code, tmp, 23)
        tmp = sform('{0:1s}', dat.seiche_code, tmp, 24)
        tmp = sform('{0:1s}', dat.cultural_effects, tmp, 25)
        tmp = sform('{0:1s}', dat.unusual_events, tmp, 26)
        tmp = sform('{0:2d}', dat.max_intensity, tmp, 28, 29)
        tmp = sform('{0:1s}', dat.max_intensity_qualifier, tmp, 30)
        tmp = sform('{0:2s}', dat.intensity_scale, tmp, 31, 32)
        tmp = sform('{0:6.2f}', dat.macroseismic_latitude, tmp, 34, 39)
        tmp = sform('{0:7.2f}', dat.macroseismic_longitude, tmp, 41, 47)
        tmp = sform('{0:3.1f}', dat.macroseismic_magnitude, tmp, 49, 51)
        tmp = sform('{0:1s}', dat.type_of_magnitude, tmp, 52)
        tmp = sform('{0:4.2f}', dat.log_of_felt_area_radius, tmp, 53, 56)
        tmp = sform('{0:5.2f}', dat.log_of_area_1, tmp, 57, 61)
        tmp = sform('{0:2d}', dat.intensity_bordering_area_1, tmp, 62, 63)
        tmp = sform('{0:5.2f}', dat.log_of_area_2, tmp, 64, 68)
        tmp = sform('{0:2d}', dat.intensity_bordering_area_2, tmp, 69, 70)
        tmp = sform('{0:1s}', dat.quality_rank, tmp, 72)
        tmp = sform('{0:3s}', dat.reporting_agency, tmp, 73, 75)
        tmp = sform('{0:1s}', '2', tmp, 80)

        return tmp

    def write_record_type_3(self, tmp):
        """ Writes record type 3 - this changes depending on the preceding line
        """
        if '3' not in tmp:
            return

        tmp = ' '*80+'\n'
        tmp = sform('{0:1s}', '3', tmp, 80)

        return tmp

    def write_record_type_4(self, data):
        """ Writes record type 4"""
        if '4' not in data:
            return [', '*22]
        else:
            dat = data['4']

        self.write_record_type_7()

        tmpfin = []
        for dat in data['4']:
            tmp = str(dat.station_name)+', '
            tmp += str(dat.instrument_type)+', '
            tmp += str(dat.component)+', '
            tmp += str(dat.quality)+', '
            tmp += str(dat.phase_id)+', '
            tmp += str(dat.weighting_indicator)+', '
            tmp += str(dat.flag_auto_pick)+', '
            tmp += str(dat.first_motion)+', '
            tmp += str(dat.hour)+', '
            tmp += str(dat.minutes)+', '
            tmp += str(dat.seconds)+', '
            tmp += str(dat.duration)+', '
            tmp += str(dat.amplitude)+', '
            tmp += str(dat.period)+', '
            tmp += str(dat.direction_of_approach)+', '
            tmp += str(dat.phase_velocity)+', '
            tmp += str(dat.angle_of_incidence)+', '
            tmp += str(dat.azimuth_residual)+', '
            tmp += str(dat.travel_time_residual)+', '
            tmp += str(dat.weight)+', '
            tmp += str(dat.epicentral_distance)+', '
            tmp += str(dat.azimuth_at_source)+', '
            tmp = tmp.replace('None', '')

            tmpfin.append(tmp)

        return tmpfin

    def write_record_type_5(self, data):
        """ Writes record type 5"""
        if '5' not in data:
            return
        else:
            dat = data['5']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.text, tmp, 2, 79)
        tmp = sform('{0:1s}', '5', tmp, 80)

        return tmp

    def write_record_type_6(self, data):
        """ Writes record type 6"""
        if '6' not in data:
            return
        else:
            dat = data['6']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.tracedata_files, tmp, 2, 79)
        tmp = sform('{0:1s}', '6', tmp, 80)

        return tmp

    def write_record_type_7(self):
        """ Writes record type 7 """
        tmp = ' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO' + \
            ' AIN AR TRES W  DIS CAZ7\n'
        return tmp

    def write_record_type_e(self, data):
        """ Writes record type E"""
        if 'E' not in data:
            return ', '*8
        else:
            dat = data['E']

        if dat.latitude_error == -999 or dat.latitude_error is None:
            return ', '*8

        tmp = str(dat.gap)+', '
        tmp += str(dat.origin_time_error)+', '
        tmp += str(dat.latitude_error)+', '
        tmp += str(dat.longitude_error)+', '
        tmp += str(dat.depth_error)+', '
        tmp += str(dat.cov_xy)+', '
        tmp += str(dat.cov_xz)+', '
        tmp += str(dat.cov_yz)+', '
        tmp = tmp.replace('None', '')

        return tmp

    def write_record_type_f(self, data):
        """ Writes record type F"""

        if 'F' not in data:
            return

        for dat in data['F'].values():
            tmp = ' '*80+'\n'
            tmp = sform('{0:10.1f}', dat.strike, tmp, 1, 10)
            tmp = sform('{0:10.1f}', dat.dip, tmp, 11, 20)
            tmp = sform('{0:10.1f}', dat.rake, tmp, 21, 30)
            tmp = sform('{0:5.1f}', dat.err1, tmp, 31, 35)
            tmp = sform('{0:5.1f}', dat.err2, tmp, 36, 40)
            tmp = sform('{0:5.1f}', dat.err3, tmp, 41, 45)
            tmp = sform('{0:5.1f}', dat.fit_error, tmp, 46, 50)
            tmp = sform('{0:5.1f}', dat.station_distribution_ratio, tmp, 51,
                        55)
            tmp = sform('{0:5.1f}', dat.amplitude_ratio, tmp, 56, 60)
            tmp = sform('{0:2d}', dat.number_of_bad_polarities, tmp, 61, 62)
            tmp = sform('{0:2d}', dat.number_of_bad_amplitude_ratios, tmp, 64,
                        65)
            tmp = sform('{0:3s}', dat.agency_code, tmp, 67, 69)
            tmp = sform('{0:7s}', dat.program_used, tmp, 71, 77)
            tmp = sform('{0:1s}', dat.solution_quality, tmp, 78)
            tmp = sform('{0:1s}', 'F', tmp, 80)

        return tmp

    def write_record_type_h(self, data):
        """ Writes record type H"""

        if 'H' not in data:
            return
        else:
            dat = data['1']

        tmp = ' '*80+'\n'
        tmp = sform('{0:4d}', dat.year, tmp, 2, 5)
        tmp = sform('{0:2d}', dat.month, tmp, 7, 8)
        tmp = sform('{0:2d}', dat.day, tmp, 9, 10)
        tmp = sform('{0:1s}', dat.fixed_origin_time, tmp, 11)
        tmp = sform('{0:2d}', dat.hour, tmp, 12, 13)
        tmp = sform('{0:2d}', dat.minutes, tmp, 14, 15)
        tmp = sform('{0:6.3f}', dat.seconds, tmp, 17, 22)
        tmp = sform('{0:9.5f}', dat.latitude, tmp, 24, 32)
        tmp = sform('{0:10.5f}', dat.longitude, tmp, 34, 43)
        tmp = sform('{0:8.3f}', dat.depth, tmp, 45, 52)
        tmp = sform('{0:6.3f}', dat.depth, tmp, 54, 59)
        tmp = sform('{0:1s}', 'H', tmp, 80)

        return tmp

    def write_record_type_i(self, data):
        """ Writes record type I"""
        if 'I' not in data:
            return ', '*7
        else:
            dat = data['I']

        tmp = str(dat.last_action_done)+', '
        tmp += str(dat.date_time_of_last_action)+', '
        tmp += str(dat.operator)+', '
        tmp += str(dat.status)+', '
        tmp += str(dat.id)+', '
        tmp += str(dat.new_id_created)+', '
        tmp += str(dat.id_locked)+', '

        return tmp

    def write_record_type_m(self, data):
        """ Writes record type M"""

        if 'M' not in data:
            return
        else:
            dat = data['M']

        tmp = ' '*80+'\n'

        tmp = sform('{0:4d}', dat.year, tmp, 2, 5)
        tmp = sform('{0:2d}', dat.month, tmp, 7, 8)
        tmp = sform('{0:2d}', dat.day, tmp, 9, 10)
        tmp = sform('{0:2d}', dat.hour, tmp, 12, 13)
        tmp = sform('{0:2d}', dat.minutes, tmp, 14, 15)
        tmp = sform('{0:4.1f}', dat.seconds, tmp, 17, 20)
        tmp = sform('{0:7.3f}', dat.latitude, tmp, 24, 30)
        tmp = sform('{0:8.3f}', dat.longitude, tmp, 31, 38)
        tmp = sform('{0:5.1f}', dat.depth, tmp, 39, 43)
        tmp = sform('{0:3s}', dat.reporting_agency, tmp, 46, 48)
        tmp = sform('{0:4.1f}', dat.magnitude, tmp, 56, 59)
        tmp = sform('{0:1s}', dat.magnitude_type, tmp, 60)
        tmp = sform('{0:3s}', dat.magnitude_reporting_agency, tmp, 61, 63)
        tmp = sform('{0:7s}', dat.method_used, tmp, 71, 77)
        tmp = sform('{0:1s}', dat.quality, tmp, 78)

        tmp = sform('{0:1s}', 'M', tmp, 80)

        tmp = ' '*80+'\n'

        tmp = sform('{0:2s}', 'MT', tmp, 2, 3)
        tmp = sform('{0:6.3f}', dat.mrr_mzz, tmp, 4, 9)
        tmp = sform('{0:6.3f}', dat.mtt_mxx, tmp, 11, 16)
        tmp = sform('{0:6.3f}', dat.mpp_myy, tmp, 18, 23)
        tmp = sform('{0:6.3f}', dat.mrt_mzx, tmp, 25, 30)
        tmp = sform('{0:6.3f}', dat.mrp_mzy, tmp, 32, 37)
        tmp = sform('{0:6.3f}', dat.mtp_mxy, tmp, 39, 44)
        tmp = sform('{0:3s}', dat.reporting_agency2, tmp, 46, 48)
        tmp = sform('{0:1s}', dat.mt_coordinate_system, tmp, 49)
        tmp = sform('{0:2d}', dat.exponential, tmp, 50, 51)
        tmp = sform('{0:6.3g}', dat.scalar_moment, tmp, 53, 62)
        tmp = sform('{0:7s}', dat.method_used_2, tmp, 71, 77)
        tmp = sform('{0:1s}', dat.quality_2, tmp, 78)
        tmp = sform('{0:1s}', 'M', tmp, 80)

        return tmp

    def write_record_type_p(self, data):
        """ Writes record type P"""

        if 'P' not in data:
            return
        else:
            dat = data['P']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.filename, tmp, 2, 79)
        tmp = sform('{0:1s}', 'P', tmp, 80)

        return tmp


def sform(strform, val, tmp, col1, col2=None, nval=-999):
    """
    Formats strings

    Formats strings according with a mod for values containing the value -999
    or None. In that case it will output spaces instead. In the case of strings
    being output, they are truncated to fit the format statement. This routine
    also  puts the new strings in the correct columns

    Parameters
    ----------
    strform : python format string
        This string must be of the form {0:4.1f}, where 4.1f can be changed.
    val : float, int, str
        input value
    nval : float, int
        null value which gets substituted by spaces
    col1 : int
        start column (1 is first column)
    col2 : inr
        end column

    Returns
    -------
    tmp : str
        Output formatted string.
    """

    if col2 is None:
        col2 = col1

    slen = int(re.findall("[0-9]+", strform)[1])

    if val == nval or val is None:
        tmp2 = slen*' '
    elif 's' in strform:
        tmp2 = strform.format(val[:slen])
    else:
        tmp2 = strform.format(val)

    tmp = tmp[:col1-1]+tmp2[:slen]+tmp[col2:]
    return tmp
