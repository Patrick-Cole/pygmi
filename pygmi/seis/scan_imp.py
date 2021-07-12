# -----------------------------------------------------------------------------
# Name:        scan_imp.py (part of PyGMI)
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
"""Convert scanned bulletins to seisan format."""

import os
import re
import numpy as np
from PyQt5 import QtWidgets
import pygmi.seis.datatypes as sdt


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
        return None
    return float(inp)


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
        return None
    return int(inp)


class SIMP():
    """GUI and the program."""

    def __init__(self, parent=None):
        # PyGMI variables
        self.ifile = ''
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

# Initialize Variables
        self.datanum = -1
        self.timesection = ''
        self.datastat = []
        self.dataphase = []
        self.datahours = []
        self.datamins = []
        self.datasecs = []
        self.dataperiod = []
        self.dataamplitude = []
        self.datadist = []
        self.datad = []
        self.dataw = []
        self.data_azim = []
        self.dataresid = []
        self.mon = ''
        self.mondec = -999
        self.day = -999
        self.year = -999
        self.region = ''
        self.lat = -999
        self.laterr = -999
        self.lon = -999
        self.lonerr = -999
        self.depth = -999
        self.numstations = -999
        self.datanum = -999
        self.magnitude = -999
        self.distanceindicator = 'R'
        self.hour = '00'
        self.minute = '00'
        self.sec = '00.00'
        self.secerr = '00.00'
        self.datarms = 0
        self.ofile = None
        self.parent = parent
        self.event = {}

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
        self.parent.clearprocesslog()
        self.showprocesslog('Import Bulletin to SEISAN Format')

        ext = 'Scanned Bulletin Text File (*.txt)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent,
                                                            'Open File',
                                                            '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))

        self.ifile = str(filename)
        ifile = self.ifile

        self.showprocesslog('Input File: '+ifile)

# Read entire file
        idata = read_ifile(ifile)
        mrecs, monconv = self.get_record_info(idata)

# Start main loop
        dat = []
        for i in mrecs:
            self.reset_vars(i[0], monconv[i[0]])

            try:
                self.extract_date(idata[i[1]])  # self.day, self.year from here
                self.look_for_origin(idata[i[1]])
            except ValueError:
                self.inerror(idata[i[1]])
            currecs = idata[i[1]:i[2]]

            for j in currecs:
                try:
                    self.look_for_region(j)
                    self.look_for_latitude(j)
                    self.look_for_depth(j)
                    self.look_for_magnitude(j)
                except ValueError:
                    self.inerror(j)

            self.numstations = 0
            if i[3] == 'RECORDA':
                self.get_data_record_a(currecs)
            if i[3] == 'RECORDB':
                self.get_data_record_b(currecs)
            if i[3] == 'RECORDC':
                self.get_data_record_c(currecs)
            self.calc_rms()

# Make sure we have at least one time
            if self.hour == 0 and self.minute == 0 and self.sec == 0.0:
                self.hour = self.datahours[0]
                self.minute = self.datamins[0]
                self.sec = self.datasecs[0]

# Put data into SEISAN structures
            self.event = {}
            self.get_record_type_1()
            self.get_record_type_e()
            self.get_record_type_i()
            self.get_record_type_4()

            dat.append(self.event)

# Open output files
        ofile = ifile[:-3]+'err'
        self.ofile = open(ofile, 'w')

        self.showprocesslog('Error File: '+ofile)

        self.ofile.write(self.parent.textbrowser_processlog.toPlainText())
        self.ofile.close()

        self.outdata['Seis'] = dat
        self.showprocesslog('\nCompleted!')

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
        return False

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

#        projdata['ftype'] = '2D Mean'

        return projdata

    def reset_vars(self, mon, mondec):
        """
        Use to reset the header variables to their defaults.

        Parameters
        ----------
        mon : str
            Month.
        mondec : int
            Integer month.

        Returns
        -------
        None.

        """
        self.mon = mon
        self.mondec = mondec
        self.region = ''
        self.lat = -999
        self.laterr = -999
        self.lon = -999
        self.lonerr = -999
        self.depth = -999
        self.numstations = -999
        self.datanum = -999
        self.magnitude = -999
        self.distanceindicator = 'R'
        self.hour = 0
        self.minute = 0
        self.sec = 0.0
        self.secerr = 0.0
        self.datarms = 0

    def reset_data_vars(self):
        """
        Reset the various data variables.

        Returns
        -------
        None.

        """
        self.datanum = -1
        self.timesection = ''
        self.datastat = []
        self.dataphase = []
        self.datahours = []
        self.datamins = []
        self.datasecs = []
        self.dataperiod = []
        self.dataamplitude = []
        self.datadist = []
        self.datad = []
        self.dataw = []
        self.data_azim = []
        self.dataresid = []

    def init_data_var_row(self):
        """
        Initialize a data variable row.

        Returns
        -------
        None.

        """
        self.datastat.append('   ')
        self.dataphase.append(' ')
        self.datahours.append(0)
        self.datamins.append(0)
        self.datasecs.append(0.0)
        self.dataperiod.append(0.0)
        self.dataamplitude.append(0.0)
        self.datadist.append(0.0)
        self.datad.append(' ')
        self.dataw.append(1)
        self.data_azim.append(0.0)
        self.dataresid.append(0.0)

# Perhaps we should have a place to load in stations outside of the program?

    def extract_date(self, tmp):
        """
        Extract the date from the string.

        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        None.

        """
        tmp2 = tmp.partition(self.mon)
        self.day = int(tmp2[0])
        tmp2 = tmp2[2].split()
        self.year = int(tmp2[0])

    def look_for_magnitude(self, tmp):
        """
        Look for the magnitude in the string.

        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        None.

        """
        if tmp.find('MAGNITUDE') > -1 and tmp.find('PHASE') == -1:
            self.magnitude = float(tmp.partition('+-')[0][-4:])

    def look_for_latitude(self, tmp):
        """
        Look for latitude and longitude.

        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        None.

        """
        if tmp.find('LATITUDE') > -1:
            tmp = tmp.replace(' ', '')
# remove extra illegal characters
            tmp2 = ''
            for i in tmp:
                if i.isalnum() or i == '.' or i == '-' or i == '+' or i == '=':
                    tmp2 += i
                else:
                    tmp2 += ''
            tmp = tmp2
# extracted and forced negative
            self.lat = -1*abs(float(tmp[tmp.find('.')-2:tmp.find('.')+3]))
            tmp = tmp[tmp.find('.')+3:]
            self.laterr = float(tmp[tmp.find('.')-3:tmp.find('.')+3])
            tmp = tmp[tmp.find('.')+3:]
# extracted and forced negative
            self.lon = float(tmp[tmp.find('.')-2:tmp.find('.')+3])
            tmp = tmp[tmp.find('.')+3:]
            self.lonerr = float(tmp[tmp.find('.')-3:tmp.find('.')+3])

    def look_for_depth(self, tmp):
        """
        Look for depth.

        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        None.

        """
        if tmp.find('DEPTH') > -1:
            tmp = tmp.partition('=')[2]
            tmp = tmp.partition('km')[0]
            self.depth = float(tmp)

    def look_for_region(self, tmp):
        """
        Look for region.

        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        None.

        """
        if tmp.find('REGION') > -1:
            self.region = tmp.partition(' ')[2]

    def look_for_origin(self, tmp):
        """
        Look for origin.

        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        None.

        """
        if tmp.find('ORIGIN') > -1:
            self.hour = int(ncor(tmp.partition('h')[0][-2:]))
            self.minute = int(ncor(tmp.partition('m')[0][-2:]))
            self.sec = float(ncor(tmp.partition('+-')[0][-5:]))
            self.secerr = float(ncor(tmp.partition('s')[0][-5:]))

    def get_data_record_a(self, tmp):
        """
        Get record type A.

        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        None.

        """
        did_alt_phase = False
        self.reset_data_vars()
# Get rid of spaces
        tmp = clean_string(tmp)

        self.distanceindicator = 'D'
        for i in tmp[0:]:
            station = is_stat(i)
            if i.find('LZ') == -1:
                tmp2 = i.split()
                self.init_data_var_row()
                try:
                    tmp2 = self.get_station(tmp2, station)
                    tmp2 = self.get_data_phase(tmp2, station, '')
                    tmp2 = self.get_data_d(tmp2)
                    tmp2 = self.get_time_section(tmp2)
                    tmp2 = self.get_data_period(tmp2)
                    tmp2 = self.get_data_amplitude(tmp2)
                except ValueError:
                    self.inerror(i)
                if station == '':
                    did_alt_phase = True
        self.datanum = len(self.datastat)
        self.numstations = int(self.datanum)
        if did_alt_phase is True:
            self.numstations = int(self.numstations / 2)
        self.dataresid = [0.0]*self.datanum
        self.data_azim = [0.0]*self.datanum
        self.datadist = [0.0]*self.datanum

    def get_data_record_b(self, tmp):
        """
        Get record type B.

        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        None.

        """
        self.reset_data_vars()
# Get rid of spaces
        tmp = clean_string(tmp)

        self.distanceindicator = 'R'
        for i in tmp[0:]:
            station = is_stat(i)
            tmp2 = i.split()
            self.init_data_var_row()
            try:
                tmp2 = self.get_station(tmp2, station)
                tmp2 = self.get_data_dist(tmp2, station)
                tmp2 = self.get_data_phase(tmp2, station, 'ES')
                tmp2 = self.get_data_d(tmp2)
                tmp2 = self.get_time_section(tmp2)
                tmp2 = self.get_data_period(tmp2)
                tmp2 = self.get_data_amplitude(tmp2)
            except ValueError:
                self.inerror(i)
        self.datanum = len(self.datastat)
        self.numstations = int(self.datanum/2)
        self.dataresid = [0.0]*self.datanum
        self.data_azim = [0.0]*self.datanum

    def get_data_record_c(self, tmp):
        """
        Get record type C.

        Parameters
        ----------
        tmp : str
            Data string.

        Returns
        -------
        None.

        """
        self.reset_data_vars()
# Get rid of spaces
        tmp = clean_string(tmp)

        self.distanceindicator = 'R'
        for i in tmp[0:]:
            station = is_stat(i)
            tmp2 = i.split()
            self.init_data_var_row()
            try:
                tmp2 = self.get_station(tmp2, station)
                tmp2 = self.get_data_dist(tmp2, station)
                tmp2 = self.get_data_azim(tmp2, station)
                tmp2 = self.get_data_phase(tmp2, station, '')
                tmp2 = self.get_data_d(tmp2)
                tmp2 = self.get_time_section(tmp2)
                tmp2 = self.get_data_resid(tmp2, station)
                tmp2 = self.get_data_period(tmp2)
                tmp2 = self.get_data_amplitude(tmp2)
            except ValueError:
                self.inerror(i)
        self.datanum = len(self.datastat)
        self.numstations = int(self.datanum/2)

    def get_station(self, tmp, station):
        """
        Get the station.

        Parameters
        ----------
        tmp : list
            Data string list.
        station : str
            Station string.

        Returns
        -------
        tmp : list
            Data string list.

        """
        if not tmp:
            return tmp

        if station != '':
            self.datastat[-1] = station
            tmp.pop(0)
        elif len(self.datastat) > 1:
            self.datastat[-1] = self.datastat[-2]
        else:
            self.datastat[-1] = '???'
        return tmp

    def get_data_dist(self, tmp, station):
        """
        Get the distance.

        Parameters
        ----------
        tmp : list
            Data string list.
        station : str
            Station string.

        Returns
        -------
        tmp : list
            Data string list.

        """
        if not tmp:
            return tmp

        if station == '' and len(self.datastat) > 1:
            self.datadist[-1] = self.datadist[-2]
        elif station == '' and len(self.datastat) < 2:
            self.datadist[-1] = -999
        if (station != '' and isfloat(tmp[0]) is True and
                tmp[0].isdigit() is False):
            self.datadist[-1] = float(tmp[0])
            tmp.pop(0)
        return tmp

    def get_data_azim(self, tmp, station):
        """
        Get the azimuth.

        Parameters
        ----------
        tmp : list
            Data string list.
        station : str
            Station string.

        Returns
        -------
        tmp : list
            Data string list.

        """
        if not tmp:
            return tmp

        if station == '' and len(self.data_azim) > 1:
            self.data_azim[-1] = self.data_azim[-2]
        elif station == '' and len(self.data_azim) < 2:
            self.data_azim[-1] = -999
        if (station != '' and isfloat(tmp[0]) is True and
                tmp[0].isdigit() is False):
            self.data_azim[-1] = float(tmp[0])
            tmp.pop(0)
        return tmp

    def get_data_phase(self, tmp, station, default):
        """
        Get the phase.

        Parameters
        ----------
        tmp : list
            Data string list.
        station : str
            Station string.
        default : str
            Default data phase.

        Returns
        -------
        tmp : list
            Data string list.

        """
        if not tmp:
            return tmp

        self.dataphase[-1] = default
        if station != '':
            self.dataphase[-1] = ''
            if tmp[0].find('E') != -1:
                self.dataphase[-1] += 'E'
            if tmp[0].find('I') != -1:
                self.dataphase[-1] += 'I'
            if tmp[0].find('P') != -1:
                self.dataphase[-1] += 'P'
            if tmp[0].find('S') != -1:
                self.dataphase[-1] += 'S'
            tmp.pop(0)
        elif tmp[0].find('P') != -1 or tmp[0].find('S') != -1:
            tmp.pop(0)
        return tmp

    def get_data_d(self, tmp):
        """
        Get the D value.

        Parameters
        ----------
        tmp : list
            Data string list.

        Returns
        -------
        tmp : list
            Data string list.

        """
        if not tmp:
            return tmp

        if tmp[0].isalpha():
            self.datad[-1] = tmp[0]
            tmp.pop(0)
        elif tmp[0] == '*':
            self.dataw[-1] = 4
            tmp.pop(0)
        return tmp

    def get_time_section(self, tmp):
        """
        Get the time.

        Parameters
        ----------
        tmp : list
            Data string list.

        Returns
        -------
        tmp : list
            Data string list.

        """
        if not tmp:
            return tmp
        tmp2 = tmp.pop(0)
        if tmp2.find('.') == -1:
            if not tmp:
                return tmp
            tmp2 += tmp.pop(0)
        if tmp2.find('.') == -1:
            if not tmp:
                return tmp
            tmp2 += tmp.pop(0)
        tmp2 = tmp2.zfill(8)  # pad with zeros if needed
        while tmp2.count('.') > 1:
            tmp2 = tmp2.replace('.', '', 1)

        self.datahours[-1] = int(tmp2[0:2])
        self.datamins[-1] = int(tmp2[2:4])
        self.datasecs[-1] = float(tmp2[4:])
        if self.datasecs[-1] > 60.0:
            self.showprocesslog('Possible problem with time field')
        return tmp

    def get_data_resid(self, tmp, station):
        """
        Get the data residual.

        Parameters
        ----------
        tmp : list
            Data string list.
        station : str
            Station string.

        Returns
        -------
        tmp : list
            Data string list.

        """
        if not tmp:
            return tmp
        if (station != '' and isfloat(tmp[0]) is True and
                tmp[0].isdigit() is False):
            self.dataresid[-1] = float(tmp[0])
            tmp.pop(0)
        return tmp

    def get_data_period(self, tmp):
        """
        Get the period.

        Parameters
        ----------
        tmp : list
            Data string list.

        Returns
        -------
        tmp : list
            Data string list.

        """
        if tmp:
            self.dataperiod[-1] = float(tmp[0])
            tmp.pop(0)
        return tmp

    def get_data_amplitude(self, tmp):
        """
        Get the amplitude.

        Parameters
        ----------
        tmp : list
            Data string list.

        Returns
        -------
        tmp : list
            Data string list.

        """
        if tmp:
            self.dataamplitude[-1] = float(tmp[0][:4])*1000
            tmp.pop(0)
        return tmp

    def calc_rms(self):
        """
        Calculate the RMS.

        Returns
        -------
        None.

        """
        rmscnt = 0
        if self.datanum > -1:
            for i in range(self.datanum):
                if self.dataw[i] != 4 and abs(self.dataresid[i]) > 0:
                    self.datarms += self.dataresid[i]**2
                    rmscnt = rmscnt + 1
            if rmscnt > 0:
                self.datarms = self.datarms / rmscnt
                self.datarms = np.sqrt(self.datarms)
            else:
                self.datarms = 0

    def get_record_type_1(self):
        """
        Write record type 1.

        Returns
        -------
        None.

        """
        tmp = sdt.seisan_1()
        tmp.year = self.year
        tmp.month = self.mondec
        tmp.day = self.day
        tmp.hour = self.hour
        tmp.minutes = self.minute
        tmp.seconds = self.sec
        tmp.distance_indicator = self.distanceindicator
        tmp.latitude = self.lat  # -999 means none
        tmp.longitude = self.lon  # -999 means none
        tmp.depth = self.depth  # -999 means none
        if self.depth != -999:
            tmp.depth_indicator = 'F'
        tmp.hypocenter_reporting_agency = 'PRE'
        tmp.number_of_stations_used = self.numstations
        tmp.rms_of_time_residuals = self.datarms
        tmp.magnitude_1 = self.magnitude
        tmp.type_of_magnitude_1 = 'L'
        tmp.magnitude_reporting_agency_1 = 'PRE'
        self.event['1'] = tmp

    def get_record_type_e(self):
        """
        Get record type E.

        Returns
        -------
        None.

        """
        if self.laterr != -999:
            tmp = sdt.seisan_1()
            tmp.gap = 0
            tmp.origin_time_error = self.secerr
            tmp.latitude_error = self.laterr
            tmp.longitude_error = self.lonerr
            tmp.depth_error = 0
            tmp.cov_xy = 0
            tmp.cov_xz = 0
            tmp.cov_yz = 0

            self.event['E'] = tmp

    def get_record_type_i(self):
        """
        Get record type I.

        Returns
        -------
        None.

        """
        tmp = sdt.seisan_I()

        tmp.last_action_done = 'SPL'
        tmp.date_time_of_last_action = '09-02-11 11:35'
        tmp.operator = 'ian '
        tmp.status = ' '
        tmp.id = (str(self.year).zfill(4)+str(self.mondec).zfill(2) +
                  str(self.day).zfill(2)+str(self.hour).zfill(2) +
                  str(self.minute).zfill(2)+str(int(round(self.sec))))
        tmp.new_id_created = ' '
        tmp.id_locked = 'L'

        if self.region != '':
            tmp.region = self.region

        self.event['I'] = tmp

    def get_record_type_4(self):
        """
        Get record type 4.

        Returns
        -------
        None.

        """
        tmp2 = []
        for i in range(self.datanum):
            tmp = sdt.seisan_4()

            tmp.station_name = self.datastat[i]
            tmp.instrument_type = 'S'
            tmp.component = 'Z'
            if self.dataphase[i] != '':
                tmp.quality = self.dataphase[i][0]
                tmp.phase_id = self.dataphase[i][1:]
            if self.dataw[i] != 1:
                tmp.weighting_indicator = self.dataw[i]
            tmp.first_motion = self.datad[i]
            if (self.datahours[i] != 0 or self.datamins[i] != 0 or
                    self.datasecs[i] != 0):
                tmp.hour = self.datahours[i]
                tmp.minutes = self.datamins[i]
                tmp.seconds = self.datasecs[i]
            tmp.coda = None

            if self.dataamplitude[i] != 0:
                tmp.amplitude = self.dataamplitude[i]

            if self.dataperiod[i] != 0:
                tmp.period = self.dataperiod[i]

            tmp.direction_of_approach = None
            tmp.phase_velocity = None
            tmp.angle_of_incidence = None
            tmp.azimuth_residual = None
            tmp.travel_time_residual = None

            if self.dataresid[i] != 0:
                tmp.time_residual = self.dataresid[i]

            if self.datadist[i] != 0:
                tmp.epicentral_distance = self.datadist[i]

            if self.data_azim[i] != 0:
                tmp.azimuth_at_source = int(self.data_azim[i])

            tmp2.append(tmp)

        self.event['4'] = tmp2

    def inerror(self, eline):
        """
        Write an error to the error file.

        Parameters
        ----------
        eline : str
            String containing error.

        Returns
        -------
        None.

        """
        if eline[0:3].isalpha():
            station = is_stat(eline)
            if station == '':
                self.showprocesslog('\nStation does not exist:')
        self.showprocesslog('\nCharacter recognition error:')
        self.showprocesslog('Line: "'+eline)

    def get_record_info(self, idata):
        """
        Get the information on when records start and finish etc.

        Parameters
        ----------
        idata : list
            Input data strings.

        Returns
        -------
        mrecs : list
            List of relevant records.
        monconv : dictionary
            Dictionary to convert string months to numbers.

        """
    # Get Dates and the beginnings of sections
        monconv = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                   'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11,
                   'DEC': 12}
        mrecs = []
        for i, idatai in enumerate(idata):
            for j in monconv:
                if idatai.find(j) > -1:
                    mrecs.append([j, i])

    # Add the end of each record.
        for i in range(len(mrecs)-1):
            mrecs[i].append(mrecs[i+1][1])
        mrecs[-1].append(len(idata))

    # Add the data record type
        for i, mrecsi in enumerate(mrecs):
            dtype = ''
            for j in idata[mrecsi[1]:mrecsi[2]]:
                if j.find('STA') > -1 and j.find('INST') > -1:
                    dtype = 'RECORDA'
                if j.find('STA') > -1 and j.find('MAGNITUDE') > -1:
                    dtype = 'RECORDB'
                if j.find('STA') > -1 and j.find('RESID') > -1:
                    dtype = 'RECORDC'
            mrecs[i].append(dtype)
        return mrecs, monconv


def read_ifile(ifile):
    """
    Routine to read and clean some of the input file.

    Parameters
    ----------
    ifile : str
        Input file name.

    Returns
    -------
    idata : list
        List of data strings.

    """
    # Read entire file
    with open(ifile) as inputf:
        idata = inputf.read()

# Fix bad characters
    idata = idata.replace('\xb1', '+-')
    idata = idata.replace('\xba', ' ')
    idata = idata.replace('(P)', 'P')
    idata = idata.replace('(S)', 'S')
    idata = idata.replace(',', '.')
    idata = re.sub(r'[^\x21-x7E\n]', ' ', idata)

# Split into lines
    idata = idata.splitlines()

# Strip spaces from front of lines
    idata = [i.lstrip() for i in idata]
    return idata


def isfloat(tmp):
    """
    Check if a number is a float. Can take decimal point.

    Parameters
    ----------
    tmp : str
        Number in string form.

    Returns
    -------
    bool
        True if number is a float, False otherwise..

    """
    try:
        float(tmp)
        return True
    except ValueError:
        return False


def clean_string(tmp):
    """
    Clean string of illegal characters.

    Parameters
    ----------
    tmp : list
        List of strings to be cleaned.

    Returns
    -------
    tmp2 : list
        Cleaned list of strings.

    """
    tmp = np.copy(tmp)
    tmp = tmp[tmp != '']
    tmp = tmp.tolist()
# remove header
    while tmp[0].find('STA') == -1 or (tmp[0].find('INST') == -1 and
                                       tmp[0].find('RESID') == -1 and
                                       tmp[0].find('MAGNITUDE') == -1):
        tmp.pop(0)
    tmp.pop(0)

# remove extra illegal characters
    tmp2 = []
    for j in tmp:
        tmp2.append('')
        for i in j:
            if i.isalnum() or i == '.' or i == '-':
                tmp2[-1] += i
            else:
                tmp2[-1] += ' '

# remove page numbers
    for j, tmp2j in enumerate(tmp2):
        tmp3 = tmp2j.replace(' ', '')
        if tmp3[0] == '-' and tmp3[-1] == '-' and len(tmp3) <= 5:
            tmp2[j] = ''

# remove some records
    tmp2 = np.copy(tmp2)
    tmp2 = tmp2[tmp2 != '']
    tmp2 = tmp2[tmp2 != ' ']
    tmp2 = tmp2[tmp2 != '- ']
    tmp2 = tmp2[tmp2 != ' -']
    tmp2 = tmp2[tmp2 != '-']
    tmp2 = tmp2.tolist()

    return tmp2


def is_stat(tmp):
    """
    Check if the string is a station.

    Parameters
    ----------
    tmp : str
        String to check.

    Returns
    -------
    station : str
        station identified.

    """
    station = ''
    stationlist = ['SNA', 'BEW', 'SNA', 'BEW', 'BPI', 'BFT', 'BLE', 'BLF',
                   'BOSA', 'BFS', 'CVN', 'CER', 'ERS', 'FRS', 'GRM', 'HVD',
                   'KSR', 'KSD', 'MSN', 'NWL', 'PRY', 'PHA', 'POF', 'POG',
                   'PKA', 'SLR', 'SEK', 'SOE', 'SBO', 'SUR', 'SWZ', 'UPI',
                   'WIN', 'KTS', 'KT1', 'KT2', 'KT3', 'KIM', 'TUH', 'CAV',
                   'BUL', 'BOS', 'SME', 'CAV', 'NCA', 'SPB', 'JOZ', 'EVA',
                   'KIM', 'VIR', 'BPI', 'UMT', 'ZOM', 'LLO', 'MZU', 'VPT',
                   'CGY', 'PRE', 'MDN', 'MDL', 'PSD', 'QUA']

    for i in stationlist:
        if tmp.find(i) > -1:
            station = i

# This section checked if any stations have mis-identifies P's or F's
    if station == '':
        tmp2 = tmp.replace('P', 'F')
        for i in stationlist:
            if tmp2.find(i) > -1:
                station = i

    return station


def ncor(tmp):
    """
    Correct number problems.

    Parameters
    ----------
    tmp : str
        String to correct.

    Returns
    -------
    tmp : str
        Corrected string.

    """
    tmp = tmp.replace('I', '1')
    return tmp
