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
"""Import Seismology Data."""

import os
import re
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
import geopandas as gpd

import pygmi.seis.datatypes as sdt
from pygmi import menu_default
from pygmi.misc import ContextModule, BasicModule


def sform(strform, val, tmp, col1, col2=None, nval=-999):
    """
    Format strings.

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
    tmp : str
        Input string
    col1 : int
        start column (1 is first column)
    col2 : int
        end column. The default is None.
    nval : float, int
        null value which gets substituted by spaces. The default is -999.

    Returns
    -------
    tmp : str
        Output formatted string.
    """
    if col2 is None:
        col2 = col1

    slen = int(re.findall('[0-9]+', strform)[1])

    if val == nval or val is None:
        tmp2 = slen*' '
    elif 's' in strform:
        tmp2 = strform.format(val[:slen])
    elif np.isnan(val):
        tmp2 = slen*' '
    else:
        tmp2 = strform.format(val)

    if len(tmp2) > slen and 'e+0' in tmp2:
        tmp2 = tmp2.replace('e+0', 'e')

    if len(tmp2) > slen and 'e+' in tmp2:
        tmp2 = tmp2.replace('e+', 'e')

    tmp = tmp[:col1-1]+tmp2[:slen]+tmp[col2:]

    return tmp


def str2float(inp):
    """
    Convert a set number of columns to float, or returns None.

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
        return np.nan

    fval = float(inp)
    if abs(fval) == 99.99 or abs(fval) == 999.9:
        fval = np.nan

    return fval


def str2int(inp):
    """
    Convert a set number of columns to integer, or returns None.

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
        return np.nan
    return int(inp)


class ImportSeisan(BasicModule):
    """Import SEISAN Data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.is_import = True
        idir = os.path.dirname(os.path.realpath(__file__))
        self.tfile = os.path.join(idir, r'descriptions.txt')

        with open(self.tfile, encoding='utf-8') as inp:
            self.rnames = inp.read()

        self.rnames = self.rnames.split('\n')

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
            ext = \
                'SEISAN Format (*.out);;' +\
                'All Files (*.*)'
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
            if self.ifile == '':
                return False

        os.chdir(os.path.dirname(self.ifile))

        with open(self.ifile, encoding='utf-8') as pntfile:
            ltmp = pntfile.readlines()

        if len(ltmp[0]) < 80:
            self.showlog('Error: Problem with file')
            return False

        # This constructs a dictionary of functions
        read_record_type = {}
        read_record_type['1'] = read_record_type_1
        read_record_type['2'] = read_record_type_2
        read_record_type['3'] = read_record_type_3
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

        file_errors = []

        for iii, i in enumerate(ltmp):
            if i.strip() == '':
                if event:
                    dat.append(event)
                    event = {}
                    event['4'] = []
                    event['F'] = {}
                continue

            # Fix short lines
            if len(i) < 81:
                i = i[:-1].ljust(80)+'\n'

            ltype = i[79]

            if ltype in '67':
                continue

            if ltype == ' ' and 'PRE' in i:
                ltype = '1'
            elif ltype == ' ':
                ltype = '4'

            if ltype == '1' and event.get('1') is not None:
                continue

            try:
                tmp = read_record_type[ltype](i)
            except KeyError:
                errs = ['Error: Invalid line type: ' + str(ltype) +
                        ' on line '+str(iii+1), i]
                file_errors.append(errs)
                continue
            except ValueError:
                errs = ['Error: Problem on line: '+str(iii+1), i]
                file_errors.append(errs)
                continue

            if ltype == '1' and (np.isnan(tmp.latitude) or
                                 np.isnan(tmp.longitude)):
                errs = ['Warning: Incomplete data on line: '+str(iii+1), i]
                file_errors.append(errs)

            if ltype == 'F':
                event[ltype].update(tmp)
            elif ltype in ('4', ' '):
                ltype = '4'
                event[ltype].append(tmp)
            elif ltype == 'M' and event.get('M') is not None:
                event[ltype] = merge_m(event[ltype], tmp)
            elif ltype == '3':
                if tmp.region != '':
                    event[ltype] = tmp
                    if tmp.region not in self.rnames:
                        errs = ['Warning: Possible spelling error on on '
                                'line: '+str(iii+1)+'. Make sure the region '
                                'spelling, case and punctuation '
                                'matches exactly the definitions '
                                'in '+self.tfile, i]
                        file_errors.append(errs)
            else:
                event[ltype] = tmp

            # IP errors
            if ltype == '4' and tmp.quality == 'I':
                if tmp.phase_id[0] == 'S':
                    errs = ['Warning: IP error on line: '+str(iii+1), i]
                    file_errors.append(errs)
                elif (tmp.phase_id[0] == 'P' and
                      tmp.first_motion not in ['C', 'D']):
                    errs = ['Warning: IP error (first motion must be C or D)'
                            ' on line: '+str(iii+1), i]
                    file_errors.append(errs)

            # EP/S phase errors
            if ltype == '4' and tmp.quality == 'E':
                if tmp.first_motion in ['C', 'D']:
                    errs = [r'Warning: EP/S error (first motion must be empty)'
                            ' on line: '+str(iii+1), i]
                    file_errors.append(errs)

            # High time residuals
            if ltype == '4' and tmp.quality == 'E':
                if tmp.travel_time_residual > 3:
                    errs = [r'Warning: Travel time residual > 3 on '
                            'line: '+str(iii+1), i]
                    file_errors.append(errs)

            if ltype == '4' and len(event['4']) > 1:
                dat1 = event['4'][-2]
                dat2 = event['4'][-1]
                if dat1.station_name == dat2.station_name:
                    if 'AML' in dat1.phase_id and dat2.phase_id[0] != ' ':
                        errs = [r'Warning: Phases may be out of order on '
                                'line: '+str(iii+1), i]
                        file_errors.append(errs)

        has_errors = any('Error' in s for s in file_errors)

        if file_errors:
            if has_errors is False:
                self.showlog('Warning: Problem with file')
                self.showlog('Process will continue, but please '
                             'see warnings in '+self.ifile+'.log')
            else:
                self.showlog('Error: Problem with file')
                self.showlog('Process stopping, please see errors '
                             'in '+self.ifile+'.log')
            with open(self.ifile+'.log', 'w', encoding='utf-8') as fout:
                for i in file_errors:
                    fout.write(i[0]+'\n')
                    fout.write(i[1]+'\n')

            if has_errors is True:
                return False
        else:
            self.showlog('No errors in the file')

        if event:
            dat.append(event)

        self.outdata['Seis'] = dat
        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)



def read_record_type_1(i):
    """
    Read record type 1.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_1
        SEISAN 1 record.

    """
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

    # make sure coordinates have the correct range
    tmp.longitude = ((tmp.longitude - 180) % 360) - 180
    tmp.latitude = ((tmp.latitude - 90) % 180) - 90

    return tmp


def read_record_type_2(i):
    """
    Read record type 2.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_2
        SEISAN 2 record.

    """
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


def read_record_type_3(i):
    """
    Read record type 3.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_4
        SEISAN 4 record.

    """
    dat = sdt.seisan_3()
    dat.text = i
    if 'Region:' in i:
        dat.region = i[12:-2].strip()

    return dat


def read_record_type_4(i):
    """
    Read record type 4.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_4
        SEISAN 4 record.

    """
    tmp = sdt.seisan_4()

    tmp.station_name = i[1:6]
    tmp.instrument_type = i[6]
    tmp.component = i[7]
    tmp.quality = i[9]
    tmp.phase_id = i[10:14]

    tmp.weighting_indicator = np.nan
    if i[14] != '_':
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
    tmp.travel_time_residual = np.nan
    if '*' not in i[63:68]:
        tmp.travel_time_residual = str2float(i[63:68])
    tmp.weight = str2int(i[68:70])
    tmp.epicentral_distance = str2float(i[70:75])
    tmp.azimuth_at_source = str2int(i[76:79])

    return tmp


def read_record_type_5(i):
    """
    Read record type 5.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_5
        SEISAN 5 record.

    """
    tmp = sdt.seisan_5()
    tmp.text = i[1:79]
    return tmp


def read_record_type_6(i):
    """
    Read record type 6.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_6
        SEISAN 6 record.

    """
    tmp = sdt.seisan_6()
    tmp.tracedata_files = i[1:79]

    return tmp


def read_record_type_e(i):
    """
    Read record type E.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_E
        SEISAN E record.

    """
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
    """
    Read record type F.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    out : dictionary
        Dictionary with a SEISAN F record.

    """
    tmp = sdt.seisan_F()

    tmp.program_used = i[70:77]
    prg = tmp.program_used.strip()
    tmp.strike = str2float(i[0:10])
    tmp.dip = str2float(i[10:20])
    tmp.rake = str2float(i[20:30])
    if prg in ('FPFIT', 'HASH'):
        tmp.err1 = str2float(i[30:35])
        tmp.err2 = str2float(i[35:40])
        tmp.err3 = str2float(i[40:45])
        tmp.fit_error = str2float(i[45:50])
        tmp.station_distribution_ratio = str2float(i[50:55])
    if prg in ('FOCMEC', 'HASH'):
        tmp.amplitude_ratio = str2float(i[55:60])
    tmp.agency_code = i[66:69]
    tmp.solution_quality = i[77]

    if prg == '':
        out = {'OTHER': tmp}
    else:
        out = {prg: tmp}

    return out


def read_record_type_h(i):
    """
    Read record type H.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_H
        SEISAN H record.

    """
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
    """
    Read record type I.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_I
        SEISAN I record.

    """
    tmp = sdt.seisan_I()

    tmp.last_action_done = i[8:11]
    tmp.date_time_of_last_action = i[12:26]
    tmp.operator = i[30:35]
    tmp.status = i[42:57]
    tmp.id = i[60:74]
    tmp.new_id_created = i[74]
    tmp.id_locked = i[75]

    return tmp


def read_record_type_m(i):
    """
    Read record type M.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_M
        SEISAN M record.

    """
    if i[1:3] != 'MT':
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
        tmp = sdt.seisan_M()
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


def merge_m(rec1, rec2):
    """
    Merge M records.

    Parameters
    ----------
    rec1 : sdt.seisan_M
        SEISAN M record.
    rec2 : sdt.seisan_M
        SEISAN M record.

    Returns
    -------
    rec1 : sdt.seisan_M
        SEISAN M record.

    """
    rec1.mrr_mzz = rec2.mrr_mzz
    rec1.mtt_mxx = rec2.mtt_mxx
    rec1.mpp_myy = rec2.mpp_myy
    rec1.mrt_mzx = rec2.mrt_mzx
    rec1.mrp_mzy = rec2.mrp_mzy
    rec1.mtp_mxy = rec2.mtp_mxy
    rec1.reporting_agency2 = rec2.reporting_agency2
    rec1.mt_coordinate_system = rec2.mt_coordinate_system
    rec1.exponential = rec2.exponential
    rec1.scalar_moment = rec2.scalar_moment
    rec1.method_used_2 = rec2.method_used_2
    rec1.quality_2 = rec2.quality_2

    return rec1


def read_record_type_p(i):
    """
    Read record type P.

    Parameters
    ----------
    i : str
        String to read from.

    Returns
    -------
    tmp : sdt.seisan_P
        SEISAN P record.

    """
    tmp = sdt.seisan_P()
    tmp.filename = i[1:79]
    return tmp


class ImportGenericFPS(BasicModule):
    """
    Import Generic Fault Plane Solution Data.

    This is stored in a csv file.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_import = True

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
            QtWidgets.QMessageBox.information(self.parent, 'File Format',
                                              'The file should have the '
                                              'following columns: '
                                              'longitude, latitude, '
                                              'depth, strike, dip, rake, '
                                              'magnitude.')

            ext = ('Comma Delimited Text (*.csv);;'
                   'All Files (*.*)')

            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
            if self.ifile == '':
                return False
        os.chdir(os.path.dirname(self.ifile))

        dlim = ','

        with open(self.ifile, encoding='utf-8') as pntfile:
            ltmp = pntfile.readline()

        isheader = any(c.isalpha() for c in ltmp)

        srows = 0
        ltmp = ltmp.split(dlim)
        if isheader:
            srows = 1
        else:
            ltmp = [str(c) for c in range(len(ltmp))]

        try:
            datatmp = np.loadtxt(self.ifile, delimiter=dlim, skiprows=srows)
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

        self.outdata['GenFPS'] = dat

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)


class ExportSeisan(ContextModule):
    """Export SEISAN Data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lmod = None
        self.fobj = None

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        if 'Seis' not in self.indata:
            self.showlog(
                'Error: You need to have a SEISAN data first!')
            return

        self.parent.process_is_active(True)

        data = self.indata['Seis']

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self.parent,
                                                            'Save File',
                                                            '.', 'out (*.out)')

        if filename == '':
            self.parent.process_is_active(False)
            return

        os.chdir(os.path.dirname(filename))

        with open(filename, 'w', encoding='utf-8') as self.fobj:
            for i in self.piter(data):
                self.write_record_type_1(i)
                self.write_record_type_f(i)  # This is missing  some #3 recs
                self.write_record_type_m(i)  # This is missing  some #3 recs
                self.write_record_type_e(i)
                self.write_record_type_i(i)  # This is missing  some #3 recs
                self.write_record_type_4(i)
                self.fobj.write(' \n')

        self.showlog('Export to Seisan Finished!')
        self.parent.process_is_active(False)

    def write_record_type_1(self, data):
        """
        Write record type 1.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if '1' not in data:
            return
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
        """
        Write record type 2.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if '2' not in data:
            return
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
        """
        Write record type 3.

        This changes depending on the preceding line.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if '3' not in tmp:
            return

        tmp = ' '*80+'\n'
        tmp = sform('{0:1s}', '3', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_4(self, data):
        """
        Write record type 4.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if '4' not in data:
            return
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
            tmp = sform('{0:>7.4g}', dat.amplitude, tmp, 34, 40, 0)
            tmp = sform('{0:4.3g}', dat.period, tmp, 42, 45, 0)
            tmp = sform('{0:5.1f}', dat.direction_of_approach, tmp, 47, 51)
            tmp = sform('{0:4.0f}', dat.phase_velocity, tmp, 53, 56)
            tmp = sform('{0:4.0f}', dat.angle_of_incidence, tmp, 57, 60)
            tmp = sform('{0:3d}', dat.azimuth_residual, tmp, 61, 63)
            tmp = sform('{0:5.2f}', dat.travel_time_residual, tmp, 64, 68, 0)
            tmp = sform('{0:2d}', dat.weight, tmp, 69, 70)
            tmp = sform('{0:5.4g}', dat.epicentral_distance, tmp, 71, 75, 0)
            tmp = sform('{0:3d}', dat.azimuth_at_source, tmp, 77, 79, 0)

            self.fobj.write(tmp)

    def write_record_type_5(self, data):
        """
        Write record type 5.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if '5' not in data:
            return
        dat = data['5']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.text, tmp, 2, 79)
        tmp = sform('{0:1s}', '5', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_6(self, data):
        """
        Write record type 6.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if '6' not in data:
            return
        dat = data['6']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.tracedata_files, tmp, 2, 79)
        tmp = sform('{0:1s}', '6', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_7(self):
        """
        Write record type 7.

        Returns
        -------
        None.

        """
        tmp = (' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO'
               ' AIN AR TRES W  DIS CAZ7\n')
        self.fobj.write(tmp)

    def write_record_type_e(self, data):
        """
        Write record type E.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if 'E' not in data:
            return
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
        """
        Write record type F.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
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
        """
        Write record type H.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if 'H' not in data:
            return

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
        """
        Write record type I.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if 'I' not in data:
            return

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
        """
        Write record type M.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if 'M' not in data:
            return

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
        tmp = sform('{0:6.3G}', dat.scalar_moment, tmp, 53, 62)
        tmp = sform('{0:7s}', dat.method_used_2, tmp, 71, 77)
        tmp = sform('{0:1s}', dat.quality_2, tmp, 78)
        tmp = sform('{0:1s}', 'M', tmp, 80)

        self.fobj.write(tmp)

    def write_record_type_p(self, data):
        """
        Write record type P.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        None.

        """
        if 'P' not in data:
            return

        dat = data['P']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.filename, tmp, 2, 79)
        tmp = sform('{0:1s}', 'P', tmp, 80)

        self.fobj.write(tmp)


class ExportCSV(ContextModule):
    """Export SEISAN Data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lmod = None
        self.fobj = None

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        if 'Seis' not in self.indata:
            self.showlog('Error: You need to have a SEISAN data first!')
            return
        self.parent.process_is_active(True)

        data = self.indata['Seis']

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self.parent,
                                                            'Save File',
                                                            '.', 'csv (*.csv)')
        if filename == '':
            self.parent.process_is_active(False)
            return
        os.chdir(os.path.dirname(filename))

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

        with open(filename, 'w', encoding='utf-8') as self.fobj:
            self.fobj.write(headi+head1+heade+head4+'\n')

            for i in self.piter(data):
                reci = self.write_record_type_i(i)
                rec1 = self.write_record_type_1(i)
                rece = self.write_record_type_e(i)
                rec4 = self.write_record_type_4(i)
                for j in rec4:
                    jtmp = reci+rec1+rece+j+'\n'
                    jtmp = jtmp.replace('nan', '')
                    self.fobj.write(jtmp)

        self.showlog('Export to csv Finished!')
        self.parent.process_is_active(False)

    def write_record_type_1(self, data):
        """
        Write record type 1.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if '1' not in data:
            return ', '*27

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
        """
        Write record type 2.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if '2' not in data:
            return None

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
        """
        Write record type 3.

        This changes depending on the preceding line.


        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        tmp : str
            Output string.

        """
        if '3' not in tmp:
            return None

        tmp = ' '*80+'\n'
        tmp = sform('{0:1s}', '3', tmp, 80)

        return tmp

    def write_record_type_4(self, data):
        """
        Write record type 4.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmpfin : list
            List of output string.

        """
        if '4' not in data:
            return [', '*22]

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
        """
        Write record type 5.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if '5' not in data:
            return None
        dat = data['5']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.text, tmp, 2, 79)
        tmp = sform('{0:1s}', '5', tmp, 80)

        return tmp

    def write_record_type_6(self, data):
        """
        Write record type 6.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if '6' not in data:
            return None
        dat = data['6']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.tracedata_files, tmp, 2, 79)
        tmp = sform('{0:1s}', '6', tmp, 80)

        return tmp

    def write_record_type_7(self):
        """
        Write record type 7.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        tmp = (' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO'
               ' AIN AR TRES W  DIS CAZ7\n')
        return tmp

    def write_record_type_e(self, data):
        """
        Write record type E.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if 'E' not in data:
            return ', '*8

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
        """
        Write record type F.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if 'F' not in data:
            return None

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
        """
        Write record type H.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if 'H' not in data:
            return None

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
        """
        Write record type I.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if 'I' not in data:
            return ', '*7

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
        """
        Write record type M.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if 'M' not in data:
            return None

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
        """
        Write record type P.

        Parameters
        ----------
        data : Dictionary
            Dictionary of record types.

        Returns
        -------
        tmp : str
            Output string.

        """
        if 'P' not in data:
            return None
        dat = data['P']

        tmp = ' '*80+'\n'
        tmp = sform('{0:78s}', dat.filename, tmp, 2, 79)
        tmp = sform('{0:1s}', 'P', tmp, 80)

        return tmp


class ExportSummary(ContextModule):
    """Export SEISAN Data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lmod = None
        self.fobj = None

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        if 'Seis' not in self.indata:
            self.showlog('Error: You need to have a SEISAN data first!')
            return
        self.parent.process_is_active(True)

        data = self.indata['Seis']

        ext = ('csv (*.csv);;'
               'Excel (*.xlsx);;'
               'Shapefile (*.shp)')

        filename, filt = QtWidgets.QFileDialog.getSaveFileName(self.parent,
                                                               'Save File',
                                                               '.', ext)
        if filename == '':
            self.parent.process_is_active(False)
            return
        os.chdir(os.path.dirname(filename))

        head = ["Year", "Month", "Day", "Hour", "Minute", "Second",
                "Latitude", "Longitude", "Depth", "Ml", "Mw", "Md", "Mb",
                "Ms", "RMS", "LatitudeError",
                "LongitudeError", "DepthError", "Mo", "SourceRadius",
                "StressDrop", "F0", "StdDeviation", "Mercalli",
                "Agency", "Description"]

        df = pd.DataFrame(columns=head)

        mtype = {}
        mtype['L'] = 'Ml'
        mtype['B'] = 'Mb'
        mtype['S'] = 'Ms'
        mtype['C'] = 'Md'
        mtype['W'] = 'Mw'

        for i, idat in enumerate(self.piter(data)):
            ML = None
            if '1' in idat:
                dat = idat['1']
                df.loc[i, 'Year'] = dat.year
                df.loc[i, 'Month'] = dat.month
                df.loc[i, 'Day'] = dat.day
                df.loc[i, 'Hour'] = dat.hour
                df.loc[i, 'Minute'] = dat.minutes
                df.loc[i, 'Second'] = dat.seconds
                df.loc[i, 'Latitude'] = dat.latitude
                df.loc[i, 'Longitude'] = dat.longitude
                df.loc[i, 'Depth'] = dat.depth

                if dat.type_of_magnitude_1 in mtype:
                    df.loc[i, mtype[dat.type_of_magnitude_1]] = dat.magnitude_1
                    if dat.type_of_magnitude_1 == 'L':
                        ML = dat.magnitude_1
                if dat.type_of_magnitude_2 in mtype:
                    df.loc[i, mtype[dat.type_of_magnitude_2]] = dat.magnitude_2
                    if dat.type_of_magnitude_2 == 'L':
                        ML = dat.magnitude_2
                if dat.type_of_magnitude_3 in mtype:
                    df.loc[i, mtype[dat.type_of_magnitude_3]] = dat.magnitude_3
                    if dat.type_of_magnitude_3 == 'L':
                        ML = dat.magnitude_3

                if ML is not None:
                    df.loc[i, 'Mercalli'] = mercalli(ML)
                df.loc[i, 'Agency'] = dat.hypocenter_reporting_agency
                df.loc[i, 'RMS'] = dat.rms_of_time_residuals

            if 'E' in idat:
                dat = idat['E']
                df.loc[i, 'LatitudeError'] = dat.latitude_error
                df.loc[i, 'LongitudeError'] = dat.longitude_error
                df.loc[i, 'DepthError'] = dat.depth_error

            if '3' in idat:
                if idat['3'] != '':
                    dat = idat['3']
                    df.loc[i, 'Description'] = dat.region

        if filt == 'csv (*.csv)':
            df.to_csv(filename, index=False)
        elif filt == 'Excel (*.xlsx)':
            df.to_excel(filename, index=False)
        elif filt == 'Shapefile (*.shp)':
            geom = gpd.points_from_xy(df.Longitude, df.Latitude)
            gdf = gpd.GeoDataFrame(df, geometry=geom)
            gdf = gdf.set_crs("EPSG:4326")
            gdf.to_file(filename)

        self.showlog('Export Summary Finished!')
        self.parent.process_is_active(False)


def mercalli(mag):
    """
    Return Mercalli index.

    Parameters
    ----------
    mag : float
        Local magnitude.

    Returns
    -------
    merc : str
        Mercalli index

    """
    if mag < 2:
        merc = 'I'
    elif mag < 3:
        merc = 'II'
    elif mag < 4:
        merc = 'III'
    elif mag == 4:
        merc = 'IV'
    elif mag < 5:
        merc = 'V'
    elif mag < 6:
        merc = 'VI'
    elif mag == 6:
        merc = 'VII'
    elif mag < 7:
        merc = 'VIII'
    elif mag == 7:
        merc = 'IX'
    elif mag < 8:
        merc = 'X'
    elif mag == 8:
        merc = 'XI'
    else:
        merc = 'XII'

    return merc


class FilterSeisan(BasicModule):
    """
    Filter Data.

    This filters data using thresholds.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.datlimits = None

        self.dsb_from = QtWidgets.QDoubleSpinBox()
        self.dsb_to = QtWidgets.QDoubleSpinBox()
        self.rectype = QtWidgets.QComboBox()
        self.recdesc = QtWidgets.QComboBox()
        self.dind = 'LRD'
        self.cb_dind_L = QtWidgets.QCheckBox('Local (L)')
        self.cb_dind_R = QtWidgets.QCheckBox('Regional (R)')
        self.cb_dind_D = QtWidgets.QCheckBox('Distant (D)')
        self.rb_rinc = QtWidgets.QRadioButton('Include')
        self.rb_rexc = QtWidgets.QRadioButton('Exclude')

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
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.datagrid')
        lbl_rectype = QtWidgets.QLabel('Record Type:')
        lbl_recdesc = QtWidgets.QLabel('Description:')
        lbl_from = QtWidgets.QLabel('From')
        lbl_to = QtWidgets.QLabel('To')
        grp_dind = QtWidgets.QGroupBox('Distance Indicator')
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.cb_dind_L)
        vbox.addWidget(self.cb_dind_R)
        vbox.addWidget(self.cb_dind_D)
        grp_dind.setLayout(vbox)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Data Filtering')
        self.cb_dind_D.setChecked(True)
        self.cb_dind_R.setChecked(True)
        self.cb_dind_L.setChecked(True)

        self.rectype.addItems(['1', '4', 'E'])
        self.recdesc.addItems(['None'])
        self.rb_rinc.setChecked(True)

        self.rectype.currentTextChanged.connect(self.rectype_init)
        self.recdesc.currentTextChanged.connect(self.recdesc_init)

        gridlayout_main.addWidget(grp_dind, 0, 0, 1, 2)
        gridlayout_main.addWidget(lbl_rectype, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.rectype, 1, 1, 1, 1)
        gridlayout_main.addWidget(lbl_recdesc, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.recdesc, 2, 1, 1, 1)
        gridlayout_main.addWidget(self.rb_rinc, 3, 0, 1, 1)
        gridlayout_main.addWidget(self.rb_rexc, 3, 1, 1, 1)
        gridlayout_main.addWidget(lbl_from, 4, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_from, 4, 1, 1, 1)
        gridlayout_main.addWidget(lbl_to, 5, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_to, 5, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.cb_dind_L.stateChanged.connect(self.dind_click)
        self.cb_dind_R.stateChanged.connect(self.dind_click)
        self.cb_dind_D.stateChanged.connect(self.dind_click)

    def dind_click(self, state):
        """
        Check checkboxes.

        Parameters
        ----------
        state : int
            State of checkbox.

        Returns
        -------
        None.

        """
        self.dind = ''
        if self.cb_dind_L.isChecked():
            self.dind += 'L'
        if self.cb_dind_R.isChecked():
            self.dind += 'R'
        if self.cb_dind_D.isChecked():
            self.dind += 'D'

        if self.dind != '':
            self.get_limits()
            self.rectype.setCurrentText('1')
            self.rectype_init('1')
        else:
            self.recdesc.disconnect()
            self.recdesc.clear()
            self.recdesc_init('')
            self.recdesc.currentTextChanged.connect(self.recdesc_init)

    def rectype_init(self, txt):
        """
        Change combo.

        Parameters
        ----------
        txt : str
            Text.

        Returns
        -------
        None.

        """
        self.rectype.disconnect()
        self.recdesc.disconnect()

        tmp = list(self.datlimits.keys())
        tmp = [i[2:] for i in tmp if i[0] == txt]

        self.recdesc.clear()
        self.recdesc.addItems(tmp)

        self.rectype.currentTextChanged.connect(self.rectype_init)
        self.recdesc.currentTextChanged.connect(self.recdesc_init)
        self.recdesc_init(self.recdesc.currentText())

    def recdesc_init(self, txt):
        """
        Change Description.

        Parameters
        ----------
        txt : str
            Text.

        Returns
        -------
        None.

        """
        if txt == '':
            minval = 0
            maxval = 0
        else:
            rectxt = self.rectype.currentText()+'_'+txt
            minval, maxval = self.datlimits[rectxt]

        self.dsb_from.setMinimum(minval)
        self.dsb_from.setMaximum(maxval)
        self.dsb_to.setMinimum(minval)
        self.dsb_to.setMaximum(maxval)
        self.dsb_from.setValue(minval)
        self.dsb_to.setValue(maxval)

    def get_limits(self):
        """
        Get limits for SEISAN data.

        Returns
        -------
        None.

        """
        dat = self.indata['Seis']
        datd = {}

        for event in dat:
            if '1' not in event:
                continue
            if event['1'].distance_indicator not in self.dind:
                continue

            allitems = [self.rectype.itemText(i)
                        for i in range(self.rectype.count())]
            for rectype in allitems:
                if rectype not in event:
                    continue
                if rectype != '4':
                    tmp = vars(event[rectype])

                    for j in tmp:
                        if isinstance(tmp[j], str):
                            continue
                        if tmp[j] is None or np.isnan(tmp[j]):
                            continue
                        newkey = rectype+'_'+j
                        if newkey not in datd:
                            datd[newkey] = []
                        datd[newkey].append(tmp[j])
                else:
                    for i in event[rectype]:
                        tmp = vars(i)
                        for j in tmp:
                            if isinstance(tmp[j], str):
                                continue
                            if tmp[j] is None or np.isnan(tmp[j]):
                                continue
                            newkey = rectype+'_'+j
                            if newkey not in datd:
                                datd[newkey] = []
                            datd[newkey].append(tmp[j])

        slist = []
        for event in datd:
            datd[event] = [min(datd[event]), max(datd[event])]
            if isinstance(datd[event][0], str):
                slist.append(event)

        for i in slist:
            del datd[i]

        self.datlimits = datd

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
        tmp = []
        if 'Seis' not in self.indata:
            return False

        # Get distance indicators
        dat = self.indata['Seis']
        dind = ''
        for event in dat:
            if '1' not in event:
                continue
            if event['1'].distance_indicator not in dind:
                dind += event['1'].distance_indicator

        self.dind = ''
        if 'L' in dind:
            self.cb_dind_L.setEnabled(True)
            self.cb_dind_L.setChecked(True)
            self.dind += 'L'
        else:
            self.cb_dind_L.setEnabled(False)
            self.cb_dind_L.setChecked(False)

        if 'R' in dind:
            self.cb_dind_R.setEnabled(True)
            self.cb_dind_R.setChecked(True)
            self.dind += 'R'
        else:
            self.cb_dind_R.setEnabled(False)
            self.cb_dind_R.setChecked(False)

        if 'D' in dind:
            self.cb_dind_D.setEnabled(True)
            self.cb_dind_D.setChecked(True)
            self.dind += 'D'
        else:
            self.cb_dind_D.setEnabled(False)
            self.cb_dind_D.setChecked(False)

        self.get_limits()
        self.rectype.setCurrentText('1')
        self.rectype_init('1')

        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return False

        self.acceptall()

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.dsb_from)
        self.saveobj(self.dsb_to)
        self.saveobj(self.rectype)
        self.saveobj(self.recdesc)
        self.saveobj(self.cb_dind_L)
        self.saveobj(self.cb_dind_R)
        self.saveobj(self.cb_dind_D)
        self.saveobj(self.rb_rinc)
        self.saveobj(self.rb_rexc)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        data = self.indata['Seis']
        rectype = self.rectype.currentText()
        recdesc = self.recdesc.currentText()
        minval = self.dsb_from.value()
        maxval = self.dsb_to.value()

        newdat = []
        for i in data:
            if '1' not in i:
                continue
            if i['1'].distance_indicator not in self.dind:
                continue
            if rectype not in i:
                continue

            if rectype != '4':
                tmp = vars(i[rectype])

                if recdesc not in tmp:
                    continue

                testval = tmp[recdesc]
                if testval is None:
                    continue

                if self.rb_rinc.isChecked() and (testval < minval or
                                              testval > maxval):
                    continue
                if not self.rb_rinc.isChecked() and (minval <= testval <= maxval):
                    continue
            else:
                for j in i[rectype]:
                    badrec = True
                    tmp = vars(j)
                    if recdesc not in tmp:
                        break

                    testval = tmp[recdesc]
                    if testval is None:
                        break

                    if self.rb_rinc.isChecked() and (testval < minval or
                                                  testval > maxval):
                        break
                    if not self.rb_rinc.isChecked() and (minval <= testval <= maxval):
                        break
                    badrec = False
                if badrec is True:
                    continue

            newdat.append(i)

        self.outdata['Seis'] = newdat
