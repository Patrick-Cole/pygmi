# -----------------------------------------------------------------------------
# Name:        usgs.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2025 Council for Geoscience
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
"""USGS SPECPR import."""

# Variable                  Description                             (bytes)
# -------------------------------------------------------------------------
# icflag          32 one bit flags:                                     4
#                 (low bit = 0, high bit = 31)
#                 bit 00 continuation data flag
#                         =0 first record of a spectrum consists of:
#                                 header then 256 data channels
#                         =1 continuation data record consisting of:
#                                 bit flags followed by 1532 bytes of
#                                 real data (bit 1=0) (383 channels)
#                                 or 1532 bytes of text (bit 1=1).
#                                 A maximum of 12 continuation records
#                                 are allowed for a total of 4852
#                                 channels (limited by arrays of 4864)
#                                 or 19860 characters of text (bit 1=1).
#                 bit 01 text/data flag:
#                        =0 the data in the array "data" is data
#                        =1 the data in the array "data" is ascii text
#                                  as is most of the header info.
#                 bit 02 flag to indicate whether or not the data for
#                        the error bar (1 sigma standard deviation of
#                        the mean) is in the next record set.
#                        =0: no errors, =1: errors in next record set.
#                 bit 03 RA, Dec / Long., Lat flag:
#                        =0 the array "ira" and "idec" corresponds to
#                           the right ascension and declination of an
#                           astronomical object.
#                        =1 the array "ira" and "idec" correspond to
#                           the longitude and latitude of a spot on a
#                           planetary surface.
#                 bit 04 variable iscta universal time/civil time flag
#                        =0 cta is civil time
#                        =1 cta is universal time
#                 bit 05 variable isctb universal time/civil time flag
#                        =0 ctb is civil time
#                        =1 ctb is universal time
#                 bit 06 unused
#                 bit 07 unused
#                 bit 08 unused
#                 bit 09 unused
#                 bit 10 unused
#                 bit 11 unused
#                 bit 12 unused
#                 bit 13 unused
#                 bit 14 unused
#                 bit 15 unused
#                 bit 16 unused
#                 bit 17 unused
#                 bit 18 unused
#                 bit 19 unused
#                 bit 20 unused
#                 bit 21 unused
#                 bit 22 unused
#                 bit 23 unused
#                 bit 24 unused
#                 bit 25 unused
#                 bit 26 unused
#                 bit 27 unused
#                 bit 28 unused
#                 bit 29 unused
#                 bit 30 unused
#                 bit 31 unused
#                 bit 32 unused

# *************** case 1: bit 00 = 0, bit 01 = 0 ****************

# icflag          Bit flags: 32 bits, see above.                        4

# ititl           40 character title which describes the               40
#                 data (ascii, character*40).

# usernm          The name of the user that created the data record     8
#                 (ascii, character*8).

# iscta           Civil or Universal time when data was                 4
#                 last processed (integer*4 in scaled seconds).
#                 Scaled by 24000.  See flag #04.

# isctb           Civil or Universal time at the start of               4
#                 the spectral run (integer*4 in scaled seconds).
#                 Scaled by 24000. See flag #05.

# jdatea          Date when data was last processed                     4
#                 Stored as integer*4 Julian Day number *10

# jdateb          Date when the spectral run began                      4
#                 Stored as integer*4 Julian Day number *10

# istb            Sidereal time when the spectral run started.          4
#                 (integer*4 in scaled seconds).
#                 Scaled by 24000. See flag #05.

# isra            Right ascension coordinates of an astronomical        4
#                 object, or longitude on a planetary surface
#                 (integer*4 numbers in seconds *1000)
#                 (RA in RA seconds, Longitude in arc-seconds)
#                 See flag #06.

# isdec           Declination coordinates of an astronomical            4
#                 object, or latitude on a planetary
#                 surface (integer*4 number in arc-seconds *1000).
#                 See flag #06.

# itchan          Total number of channels in the spectrum              4
#                 (integer*4 value from 1 to 4852)

# irmas           The equivalent atmospheric thickness through          4
#                 which the observation was obtained (=1.0
#                 overhead scaled: airmass*1000; integer*4).

# revs            The number of independent spectral scans              4
#                 which were added to make the spectrum
#                 (integer*4 number).

# iband(2)        The channel numbers which define the band             8
#                 normalization (scaling to unity). (integers*4)

# irwav           The record number within the file where the           4
#                  wavelengths are found (integer*4).

# irespt          The record pointer to where the resolution can        4
#                 be found (or horizontal error bar) (integer*4).

# irecno          The record number within the file where the           4
#                 data is located (integer*4 number).

# itpntr          Text data record pointer. This pointer points         4
#                 to a data record where additional text describing
#                 the data may be found.  (32 bit integer)

# ihist           The program automatic 60 character history.          60
#                 (ascii, character*60)

# mhist           Manual history (4 lines of 74 characters            296
#                 each.  Program automatic for large history
#                 requirements (ascii, character*296).

# nruns           The number of independent spectral runs               4
#                 which were summed or averaged to make this
#                 spectrum (integer*4).

# siangl          The angle of incidence of illuminating                4
#                radiation (Integer*4 number, in arc-seconds*6000).
#                (90 degrees=1944000000; -90 deg <= angle <= 90 deg)
#                 integrating sphere = 2000000000
#                 Geometric albedo   = 2000000001

# seangl          The angle of emission of illuminating                 4
#                 radiation (Integer*4 number, in arc-seconds*6000).
#                 (90 degrees=1944000000; -90 deg <= angle <= 90 deg)
#                 integrating sphere = 2000000000
#                 Geometric albedo   = 2000000001

# sphase          The phase angle between iangl and eangl               4
#                 (Integer*4 number, in arc-seconds*1500).
#                 (180 degrees=972000000; -180 deg <= phase <= 180 deg)
#                 integrating sphere = 2000000000

# iwtrns          Weighted number of runs (the number of runs           4
#                 of the spectrum with the minimum runs which was
#                 used in processing this spectrum, integer*4).

# itimch          The time observed in the sample beam for              4
#                 each half chop in milliseconds (for chopping
#                 spectrometers only). (integer*4)

# xnrm            The band normalization factor. For data scaled        4
#                 to 1.0, multiply by this number to recover
#                 photometric level (32 bit real number).

# scatim          The time it takes to make one scan of the             4
#                 entire spectrum in seconds (32 bit real number).

# timint          Total integration time (usually=scatime * nruns)      4
#                 (32 bit real number).

# tempd           Temperature in degrees Kelvin                         4
#                 (32 bit real number).

# data(256)       The spectral data (256 channels of 32 bit          1024
#                 real data numbers).
# -------------------------------------------------------------------------
#          case 1: total (bytes):                                    1536
# =========================================================================

# *************** case 2: bit 00  = 1, bit 01 = 0 ****************

# icflag          Flags: see case 1                                     4

# cdata(383)      The continuation of the data values (383 channels  1532
#                 of 32 bit real numbers).
# -------------------------------------------------------------------------
#          case 2: total (bytes):                                    1536
# =========================================================================

# *************** case 3: bit 00  = 0, bit 01 = 1 ****************

# icflag          Flags: see case 1                                     4

# ititle          40 character title which describes the               40
#                 data (ascii, character*40).

# usernm          The name of the user who created the data record      8
#                 (ascii, character*8).

# itxtpt          Text data record pointer. This pointer points         4
#                 to a data record where additional text may be
#                 may be found.  (32 bit integer)
# itxtch          The number of text characters (maximum= 19860).       4

# itext           1476 characters of text.  Text has embedded        1476
#                 newlines so the number of lines available is
#                 limited only by the number of characters available.
# -------------------------------------------------------------------------
#         case 3: total (bytes):                                     1536
# =========================================================================

# *************** case 4: bit 00  = 1, bit 01 = 1 ******************

# icflag          Flags: see case 1                                     4

# tdata           1532 characters of text.                            1532
# -------------------------------------------------------------------------
#          case 4: total (bytes):                                     1536
# =========================================================================

import struct

import numpy as np


def SPECPR(ifile):
    """
    SPECPR import function.

    Parameters
    ----------
    ifile : str
        Input file.

    Returns
    -------
    spec : dict
        Output spectra.

    """
    # Import data into records indexed by record number

    data = []
    with open(ifile, "rb") as f:
        _ = f.read(1536)
        while True:
            chunk = f.read(1536)
            if not chunk:
                break
            data.append(chunk)

    idata = iter(data)
    recs = {}
    rec = {}
    recnum = 0
    for dat in idata:
        recnum += 1
        orecnum = recnum
        b = unpack_icflag(dat[:4])

        if b[0] == 0 and b[1] == 0:
            rec = case1(dat)

            if rec["itchan"] > 256:
                numrecs = (rec["itchan"] - 256) // 383 + 1
                for i in range(numrecs):
                    dat = next(idata)
                    recnum += 1
                    cdata = struct.unpack(">383f", dat[4:])
                    rec["data"] += cdata
            rec["data"] = rec["data"][: rec["itchan"]]

        elif b[0] == 0 and b[1] == 1:
            rec = {}
            rec["rectype"] = 3
            rec["icflag"] = unpack_icflag(dat[:4])
            rec["ititle"] = dat[4:44].decode("latin-1")
            rec["usernm"] = dat[44:52].decode("latin-1")
            rec["itxtpt"] = struct.unpack(">i", dat[52:56])[0]
            rec["itxtch"] = struct.unpack(">i", dat[56:60])[0]
            rec["itext"] = dat[60:][: rec["itxtch"]].decode("latin-1")

            if rec["itxtch"] > 1476:
                numrecs = (rec["itxtch"] - 1476) // 1532 + 1
                for i in range(numrecs):
                    dat = next(idata)
                    recnum += 1
                    rec["itext"] += dat[4:].decode("latin-1")

            rec["itext"] = rec["itext"][: rec["itxtch"]]

            if rec["ititle"] in [
                "----------------------------------------",
                "****************************************",
            ]:
                continue

        recs[orecnum] = rec

    # Now reorganise the records
    spec = {}
    chapter = None
    for rec in recs.values():
        if "Chapter" in rec["ititle"] and ":" in rec["ititle"]:
            chapter = rec["ititle"]
        if chapter is None:
            continue
        if rec["rectype"] == 1 and "error" not in rec["ititle"].lower():
            rec["refl"] = np.array(rec["data"])
            rec["refl"][rec["refl"] <= -1.23e34] = np.nan
            rec["wvl"] = np.array(recs[rec["irwav"]]["data"]) * 1000.0
            rec["fwhm"] = recs[rec["irespt"]]["data"]
            rec["text"] = recs[rec["itpntr"]]["itext"]
            spec[rec["ititle"]] = rec

    return spec


def case1(dat):
    """
    Case 1.

    Parameters
    ----------
    dat : bytes
        Binary record.

    Returns
    -------
    rec : dict
        Output record.
    """
    rec = {}

    rec["rectype"] = 1
    rec["icflag"] = unpack_icflag(dat[:4])
    rec["ititle"] = dat[4:44].decode("latin-1")
    rec["usernm"] = dat[44:52].decode("latin-1")
    rec["iscta"] = struct.unpack(">i", dat[52:56])[0] / 24000.0
    rec["isctb"] = struct.unpack(">i", dat[56:60])[0] / 24000.0
    rec["jdatea"] = struct.unpack(">i", dat[60:64])[0] / 10.0
    rec["jdateb"] = struct.unpack(">i", dat[64:68])[0] / 10.0
    rec["istb"] = struct.unpack(">i", dat[68:72])[0] / 24000.0
    rec["isra"] = struct.unpack(">i", dat[72:76])[0] / 1000.0
    rec["isdec"] = struct.unpack(">i", dat[76:80])[0] / 1000.0
    rec["itchan"] = struct.unpack(">i", dat[80:84])[0]
    rec["irmas"] = struct.unpack(">i", dat[84:88])[0]
    rec["revs"] = struct.unpack(">i", dat[88:92])[0]
    rec["iband"] = struct.unpack(">2i", dat[92:100])
    rec["irwav"] = struct.unpack(">i", dat[100:104])[0]
    rec["irespt"] = struct.unpack(">i", dat[104:108])[0]
    rec["irecno"] = struct.unpack(">i", dat[108:112])[0]
    rec["itpntr"] = struct.unpack(">i", dat[112:116])[0]
    rec["ihist"] = dat[116:176].decode("latin-1")
    rec["mhist"] = dat[176:472].decode("latin-1")
    rec["nruns"] = struct.unpack(">i", dat[472:476])[0]
    rec["siangl"] = struct.unpack(">i", dat[476:480])[0] / 6000.0
    rec["seangl"] = struct.unpack(">i", dat[480:484])[0] / 6000.0
    rec["sphase"] = struct.unpack(">i", dat[484:488])[0] / 1500.0
    rec["iwtrns"] = struct.unpack(">i", dat[488:492])[0]
    rec["itimch"] = struct.unpack(">i", dat[492:496])[0]
    rec["xnrm"] = struct.unpack(">f", dat[496:500])[0]
    rec["scatim"] = struct.unpack(">f", dat[500:504])[0]
    rec["timint"] = struct.unpack(">f", dat[504:508])[0]
    rec["tempd"] = struct.unpack(">f", dat[508:512])[0]
    rec["data"] = struct.unpack(">256f", dat[512:])
    rec["errors"] = None

    return rec


def unpack_icflag(icflag):
    """
    Unpack the bits from icflag.

    Parameters
    ----------
    icflag : bytes
        Binary icflag.

    Returns
    -------
    b : list
        Unpacked bits.

    """
    icflag = int(icflag[3])
    b = [0, 0, 0, 0, 0, 0]

    b[0] = (icflag >> 0) & 1
    b[1] = (icflag >> 1) & 1
    b[2] = (icflag >> 2) & 1
    b[3] = (icflag >> 3) & 1
    b[4] = (icflag >> 4) & 1
    b[5] = (icflag >> 5) & 1

    return b


if __name__ == "__main__":
    SPECPR(r"D:\usgs_splib07\SPECPRsplib07\splib07a")

    print("Finished!")
