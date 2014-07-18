# -----------------------------------------------------------------------------
# Name:        ginterp.py (part of PyGMI)
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
""" This program converts scanned bulletins to seisan format """

# pylint: disable=E1101
import numpy as np
from PySide import QtGui
import os


class SIMP(object):
    """ Main form which does the GUI and the program """
    def __init__(self, parent=None):
# PyGMI variables
        self.ifile = ""
        self.name = "Import Data: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

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
        self.showtext = self.parent.showprocesslog

    def settings(self):
        """ Settings """
        self.parent.clearprocesslog()
        self.showtext('Import Bulletin to Seisan Format')

        ext = "Scanned Bulletin Text File (*.txt)"

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)[0]
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

        ifile = self.ifile

        self.showtext('Input File: '+ifile)

# Read entire file
        idata = self.read_ifile(ifile)
        mrecs, monconv = self.get_record_info(idata)

# Start main loop
        dat = ''
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

# Now write out results to file
            dat += self.write_record_type_1()
            dat += self.write_record_type_e()
            dat += self.write_record_type_i()
            dat += self.write_record_type_3()
            dat += self.write_record_type_7()
            dat += ' \n'

# Open output files
        ofile = ifile[:-3]+'sei'
        self.ofile = open(ofile, 'w')

        self.showtext('Output File: '+ofile)

        self.ofile.write(dat)
        self.ofile.close()

        ofile = ifile[:-3]+'err'
        self.ofile = open(ofile, 'w')

        self.showtext('Error File: '+ofile)

        self.ofile.write(self.parent.textbrowser_processlog.toPlainText())
        self.ofile.close()

        self.outdata['Seis'] = dat
        self.showtext('\nCompleted!')

        self.outdata['Seis'] = dat
        self.showtext('\nCompleted!')

        return True

    def reset_vars(self, mon, mondec):
        """ Used to reset the header variables to their defaults """
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
        """ Resets the various data variables """
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
        """ Initializes a data variable row """
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
        """ Extracts the date from the string """
        tmp2 = tmp.partition(self.mon)
        self.day = int(tmp2[0])
        tmp2 = tmp2[2].split()
        self.year = int(tmp2[0])

    def look_for_magnitude(self, tmp):
        """ Looks for the magnitude in the string """
        if tmp.find('MAGNITUDE') > -1 and tmp.find('PHASE') == -1:
            self.magnitude = float(tmp.partition('+-')[0][-4:])

    def look_for_latitude(self, tmp):
        """ Looks for latitude and longitude """
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
        """ Looks for depth """
        if tmp.find('DEPTH') > -1:
            tmp = tmp.partition('=')[2]
            tmp = tmp.partition('km')[0]
            self.depth = float(tmp)

    def look_for_region(self, tmp):
        """ Looks for region """
        if tmp.find('REGION') > -1:
            self.region = tmp.partition(' ')[2]

    def look_for_origin(self, tmp):
        """ Looks for origin """
        if tmp.find('ORIGIN') > -1:
            self.hour = int(self.ncor(tmp.partition('h')[0][-2:]))
            self.minute = int(self.ncor(tmp.partition('m')[0][-2:]))
            self.sec = float(self.ncor(tmp.partition('+-')[0][-5:]))
            self.secerr = float(self.ncor(tmp.partition('s')[0][-5:]))

    def ncor(self, tmp):
        """ Correct number problems """
        tmp = tmp.replace('I', '1')
        return tmp

    def get_data_record_a(self, tmp):
        """ gets record type A """
        did_alt_phase = False
        self.reset_data_vars()
# Get rid of spaces
        tmp = self.clean_string(tmp)

        self.distanceindicator = 'D'
        for i in tmp[0:]:
            station = self.is_stat(i)
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
        """ Gets record type B"""
        self.reset_data_vars()
# Get rid of spaces
        tmp = self.clean_string(tmp)

        self.distanceindicator = 'R'
        for i in tmp[0:]:
            station = self.is_stat(i)
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
        """ Gets record type C """
        self.reset_data_vars()
# Get rid of spaces
        tmp = self.clean_string(tmp)

        self.distanceindicator = 'R'
        for i in tmp[0:]:
            station = self.is_stat(i)
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
        """ Gets the station """
        if len(tmp) == 0:
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
        """ Gets the distance """
        if len(tmp) == 0:
            return tmp

        if station == '' and len(self.datastat) > 1:
            self.datadist[-1] = self.datadist[-2]
        elif station == '' and len(self.datastat) < 2:
            self.datadist[-1] = -999
        if (station != '' and self.isfloat(tmp[0]) is True and
                tmp[0].isdigit() is False):
            self.datadist[-1] = float(tmp[0])
            tmp.pop(0)
        return tmp

    def get_data_azim(self, tmp, station):
        """ Gets the azimuth """
        if len(tmp) == 0:
            return tmp

        if station == '' and len(self.data_azim) > 1:
            self.data_azim[-1] = self.data_azim[-2]
        elif station == '' and len(self.data_azim) < 2:
            self.data_azim[-1] = -999
        if (station != '' and self.isfloat(tmp[0]) is True and
                tmp[0].isdigit() is False):
            self.data_azim[-1] = float(tmp[0])
            tmp.pop(0)
        return tmp

    def get_data_phase(self, tmp, station, default):
        """ Gets the phase """
        if len(tmp) == 0:
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
        """ Gets the D value """
        if len(tmp) == 0:
            return tmp

        if tmp[0].isalpha():
            self.datad[-1] = tmp[0]
            tmp.pop(0)
        elif tmp[0] == '*':
            self.dataw[-1] = 4
            tmp.pop(0)
        return tmp

    def get_time_section(self, tmp):
        """ Gets the time """
        if len(tmp) == 0:
            return tmp
        tmp2 = tmp.pop(0)
        if tmp2.find('.') == -1:
            if len(tmp) == 0:
                return tmp
            tmp2 += tmp.pop(0)
        if tmp2.find('.') == -1:
            if len(tmp) == 0:
                return tmp
            tmp2 += tmp.pop(0)
        tmp2 = tmp2.zfill(8)  # pad with zeros if needed
        while tmp2.count('.') > 1:
            tmp2 = tmp2.replace('.', '', 1)

        self.datahours[-1] = int(tmp2[0:2])
        self.datamins[-1] = int(tmp2[2:4])
        self.datasecs[-1] = float(tmp2[4:])
        if self.datasecs[-1] > 60.0:
            self.showtext('Possible problem with time field')
#            raise NameError('Possible problem with time field')
        return tmp

    def get_data_resid(self, tmp, station):
        """ Gets the data residual """
        if len(tmp) == 0:
            return tmp
        if (station != '' and self.isfloat(tmp[0]) is True and
                tmp[0].isdigit() is False):
            self.dataresid[-1] = float(tmp[0])
            tmp.pop(0)
        return tmp

    def get_data_period(self, tmp):
        """ Gets the period """
        if len(tmp) > 0:
            self.dataperiod[-1] = float(tmp[0])
            tmp.pop(0)
        return tmp

    def get_data_amplitude(self, tmp):
        """ Gets the amplitude """
        if len(tmp) > 0:
            self.dataamplitude[-1] = float(tmp[0][:4])*1000
            tmp.pop(0)
        return tmp

    def calc_rms(self):
        """ Calculates the RMS """
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

    def write_record_type_1(self):
        """ Writes record type 1"""
# Record Type 1
# 1 Free
# 2- 5 I4 Year
# 6 Free
# 7 - 8; I2; month
# 9-10 I2 Day of Month
# 11 Fix o. time Normally blank,  an F fixes origin Time
# 12-13 I2 Hour
# 14-15 I2 Minutes
# 16 Free
# 17-20 F4.1 Seconds
# 21 Location model indicator Any character
# 22 A1 Distance Indicator L = Local,  R = Regional,  etc.
# 23 A1 Event ID E = Explosion,  etc. P = Probable explosion V = volcanic
#   Q = Probable volcanic
# 24 - 30; F7.3; Latitude; Degrees(N)
# 31-38 F8.3 Longitude Degrees (+ E)
# 39-43 F5.1 Depth Km
# 44 A1 Depth Indicator F = Fixed,  S = Starting value
# 45 A1 Locating indicator ----------------------------,  * do not locate
# 46-48 A3 Hypocenter Reporting Agency
# 49-51 Number of Stations Used
# 52-55 RMS of Time Residuals
# 56-59 F4.1 Magnitude No. 1
# 60 A1 Type of Magnitude L=ML,  B=mb,  S=Ms,  W=MW,  G=MbLg,  C=Mc
# 61-63 A3 Magnitude Reporting Agency
# 64-67 F4.1 Magnitude No. 2
# 68 A1 Type of Magnitude
# 69-71 A3 Magnitude Reporting Agency
# 72-75 F4.1 Magnitude No. 3
# 76 A1 Type of Magnitude
# 77-79 A3 Magnitude Reporting Agency
# 80 A1 Type of this line ('1'),  can be blank if first

        tmp = ' {0:4d} {0:2d}{0:2d} '.format(self.year, self.mondec, self.day)
        tmp += str(self.hour).zfill(2)
        tmp += str(self.minute).zfill(2)
        tmp += ' {0:4.1f} '.format(self.sec)
        tmp += self.distanceindicator
        tmp += ' '

        if self.lat != -999:
            tmp += '{0:7.3f}'.format(self.lat)
        else:
            tmp += ' '*7

        if self.lon != -999:
            tmp += '{0:8.3f}'.format(self.lon)
        else:
            tmp += ' '*8

        if self.depth != -999:
            tmp += '{0:5.1f}F'.format(self.depth)
        else:
            tmp += ' '*6

        tmp += ' PRE'

        if self.numstations != -999:
            tmp += '{0:3d}'.format(self.numstations)
        else:
            tmp += '  1'

        if self.datarms > 0:
            tmp += '{0:4.1f}'.format(self.datarms)
        else:
            tmp += ' '*4

        if self.magnitude != -999:
            tmp += '{0:4.1f}'.format(self.magnitude)
            tmp += 'LPRE'
        else:
            tmp += ' '*8

        tmp += ' '*16
        tmp += '1\n'

        return tmp
#        self.ofile.write(tmp)

    def write_record_type_e(self):
        """ Writes record type E"""
# Type E Line (Optional): Hyp error estimates
# 1 Free
# 2 - 5 A4 The text GAP=
# 6 - 8 I3 Gap
# 15-20 F6.2 Origin time error
# 25-30 F6.1 Latitude (y) error
# 31-32 Free
# 33-38 F6.1 Longitude (x) error (km)
# 39-43 F5.1 Depth (z) error (km)
# 44-55 E12.4 Covariance (x, y) km*km
# 56-67 E12.4 Covarience (x, z) km*km
# 68-79 E14.4 Covariance (y, z) km*km

        tmp = ''
        if self.laterr != -999:
            tmp += ' '
            tmp += 'GAP='
            tmp += '000'
            tmp += ' '*6
            tmp += '{0:6.2f}'.format(float(self.secerr))
            tmp += ' '*4
            tmp += '{0:6.1f}'.format(self.laterr)
            tmp += ' '*2
            tmp += '{0:6.1f}'.format(self.lonerr)
            tmp += '  0.0'
            tmp += '  0.0000E+00'
            tmp += '  0.0000E+00'
            tmp += '  0.0000E+00'
            tmp += 'E\n'

        return tmp
#            self.ofile.write(tmp)

    def write_record_type_i(self):
        """ Writes record type I"""
#    Type I Line,  ID line
# 1 Free
# 2:8 Help text for the action indicator
# 9:11 Last action done,  so far defined:
# SPL: Split
# REG: Register
# ARG: AUTO Register,  AUTOREG
# UPD: Update
# UP : Update only from EEV
# REE: Register from EEV
# DUB: Duplicated event
# NEW: New event
# 12 Free
# 13:26 Date and time of last action
# 27 Free
# 28:30 Help text for operator
# 31:34 Operater code
# 35 Free
# 36:42 Help text for status
# 43:56 Status flags,  not yet defined
# 57 Free
# 58:60 Help text for ID
# 61:74 ID,  year to second
# 75 if d,  this indicate that a new file id had to be created which was one or
#      more seconds different from an existing ID to avoid overwrite.
# 76 Indicate if ID is locked. Blank means not locked,  L means locked.

        tmp = ''
        tmp += ' '
        tmp += 'ACTION:'
        tmp += 'SPL'
        tmp += ' '
        tmp += '09-02-11 11:35'
        tmp += ' '
        tmp += 'OP:'
        tmp += 'ian '
        tmp += ' '
        tmp += 'STATUS:'
        tmp += '              '
        tmp += ' '
        tmp += 'ID:'
        tmp += str(self.year).zfill(4)+str(self.mondec).zfill(2) +    \
            str(self.day).zfill(2)+str(self.hour).zfill(2) +         \
            str(self.minute).zfill(2)+str(int(round(self.sec)))
        tmp += ' '
        tmp += 'L'
        tmp += '   I\n'
        return tmp
#        self.ofile.write(tmp)

    def write_record_type_3(self):
        """ Writes record type 3 """
# Type 3 Line (Optional):
# 1 Free
# 2-79 A Text Anything
# 80 A1 Type of this line ("3")

        tmp = ''
        if len(self.region) > 0:
            tmp += ' '
            tmp += 'Bul:Region: '
            tmp += self.region
            tmp += ' '*(78-len(self.region)-12)
            tmp += '3\n'
#            self.ofile.write(tmp)
        return tmp

    # Type 7 line
    def write_record_type_7(self):
        """ Writes record type 7 """
        tmp = ' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO' + \
            ' AIN AR TRES W  DIS CAZ7\n'
#        self.ofile.write(tmp)
        for i in range(self.datanum):
            tmp += ' {0:4s}'.format(self.datastat[i])
            tmp += ' SZ {0:5s} '.format(self.dataphase[i])
            if self.dataw[i] != 1:
                tmp = tmp[:-1]+'{0:1d}'.format(self.dataw[i])
            tmp += ' ' + self.datad[i] + ' '*11
            if self.datahours[i] != 0 or self.datamins[i] != 0 or   \
                    self.datasecs[i] != 0:
                tmp = tmp[:-9]+'{0:02d}{0:02d}{0:5.2f}'.format(
                    self.datahours[i], self.datamins[i], self.datasecs[i])
            tmp += ' '*12
            if self.dataamplitude[i] != 0 and self.dataamplitude[i] < 10000:
                tmp = tmp[:-6]+'{0:6.1f}'.format(self.dataamplitude[i])
            elif self.dataamplitude[i] != 0:
                tmp = tmp[:-6]+'{0:6.0f}'.format(self.dataamplitude[i])
            tmp += ' '*5
            if self.dataperiod[i] != 0:
                tmp = tmp[:-5]+' {0:4.2f}'.format(self.dataperiod[i])
            tmp += ' '*23
            if abs(self.dataresid[i]) < 10 and self.dataresid[i] != 0:
                tmp = tmp[:-5]+'{0:5.2f}'.format(self.dataresid[i])
            elif self.dataresid[i] != 0:
                tmp = tmp[:-5]+'{0:5.1f}'.format(self.dataresid[i])
            tmp += ' '*7
            if self.datadist[i] < 100 and self.datadist[i] > 0:
                tmp = tmp[:-4]+'{0:4.1f}'.format(self.datadist[i])
            elif self.datadist[i] != 0:
                tmp = tmp[:-4]+'{0:4.0f}'.format(self.datadist[i], 4, 0)
            tmp += ' '*4
            if self.data_azim[i] != 0:
                tmp = tmp[:-3]+' {0:3.0f}'.format(self.data_azim[i])
            tmp += '\n'
#            self.ofile.write(tmp)
        return tmp

    def inerror(self, eline):
        """ Writes an error to the error file """
        if eline[0:3].isalpha():
            station = self.is_stat(eline)
            if station == '':
                self.showtext('\nStation does not exist:')
        self.showtext('\nCharacter recognition error:')
        self.showtext('Line: "'+eline)

    def get_record_info(self, idata):
        """ Get the information on when records start and finish etc """
    # Get Dates and the beginnings of sections
        monconv = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                   'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11,
                   'DEC': 12}
        mrecs = []
        for i in range(len(idata)):
            for j in monconv.keys():
                if idata[i].find(j) > -1:
                    mrecs.append([j, i])

    # Add the end of each record.
        for i in range(len(mrecs)-1):
            mrecs[i].append(mrecs[i+1][1])
        mrecs[-1].append(len(idata))

    # Add the data record type
        for i in range(len(mrecs)):
            dtype = ''
            for j in idata[mrecs[i][1]:mrecs[i][2]]:
                if j.find('STA') > -1 and j.find('INST') > -1:
                    dtype = 'RECORDA'
                if j.find('STA') > -1 and j.find('MAGNITUDE') > -1:
                    dtype = 'RECORDB'
                if j.find('STA') > -1 and j.find('RESID') > -1:
                    dtype = 'RECORDC'
            mrecs[i].append(dtype)
        return mrecs, monconv

    def read_ifile(self, ifile):
        """ Routine to read and clean some of the input file """
        # Read entire file
        inputf = open(ifile)
        idata = inputf.read()

    # Fix bad characters
        idata = idata.replace('\xb1', '+-')
        idata = idata.replace('\xba', ' ')
        idata = idata.replace('(P)', 'P')
        idata = idata.replace('(S)', 'S')
        idata = idata.replace(',', '.')
        idata = self.nohex(idata)
        inputf.close()

    # Split into lines
        idata = idata.splitlines()

    # Strip spaces from front of lines
        idata = [i.lstrip() for i in idata]
        return idata

    def is_stat(self, tmp):
        """Check if the string is a station """
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

    def clean_string(self, tmp):
        """ Cleans string of illegal characters """
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
        for j in range(len(tmp2)):
            tmp3 = tmp2[j].replace(' ', '')
            if tmp3[0] == '-' and tmp3[-1] == '-'and len(tmp3) <= 5:
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

    def isfloat(self, tmp):
        """Check if a number is a float. Can take decimal point"""
        try:
            float(tmp)
            return True
        except ValueError:
            return False

    def nohex(self, tmp):
        """Get rid of hex characters in a string"""
        tmp = repr(tmp)
        while tmp.find('\\x') > -1:
            hpos = tmp.find('\\x')
            tmp = tmp[:hpos]+' '+tmp[hpos+4:]
        tmp = eval(tmp)
#        tmp = tmp.encode('unicode-escape')
#
#        while tmp.find('\\x') > -1:
#            hpos = tmp.find('\\x')
#            tmp = tmp[:hpos]+' '+tmp[hpos+4:]
#        tmp = tmp.decode('unicode-escape')

        return tmp
