# -----------------------------------------------------------------------------
# Name:        datatypes.py (part of PyGMI)
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
Class for seismic data types.

Docstring information is from SEISAN documentation.
"""


class seisan_1():
    """
    Class to hold Seisan Type 1 data.

    Type 1 Line Format:

    Columns Format Description                    Comments
    1              Free
    2-5      I4    Year
    6              Free
    7-8      I2    Month
    9-10     I2    Day of Month
    11             Fix o. time                    Normally blank, an F fixes
    -                                             origin time
    12-13    I2    Hour
    14-15    I2    Minutes
    16             Free
    17-20    F4.1  Seconds
    21             Location model indicator       Any character
    22       A1    Distance Indicator             L = Local, R = Regional,
    -                                             D = Distant, etc.
    23       A1    Event ID                       E = Confirmed explosion
    -                                             P = Probable explosion
    -                                             V = Volcanic
    -                                             Q = Confirmed earthquake
    -                                             ' ' = Presumed earthquake
    -                                             X = Landslide
    24-30    F7.3  Latitude                       Degrees (+ N)
    31-38    F8.3  Longitude                      Degrees (+ E)
    39-43    F5.1  Depth                          Km
    44       A1    Depth Indicator                F = Fixed, S = Starting value
    45       A1    Locating indicator             ----------------------------,
    -                                             * do not locate
    46-48    A3    Hypocenter Reporting Agency
    49-51          Number of Stations Used
    52-55          RMS of Time Residuals
    56-59   F4.1   Magnitude No. 1
    60 A1          Type of Magnitude L=ML, b=mb, B=mB, s=Ms, S=MS, W=MW,
    -                                G=MbLg (not used by SEISAN), C=Mc
    61-63   A3     Magnitude Reporting Agency
    64-67   F4.1   Magnitude No. 2
    68 A1          Type of Magnitude
    69-71   A3     Magnitude Reporting Agency
    72-75   F4.1   Magnitude No. 3
    76 A1          Type of Magnitude
    77-79   A3     Magnitude Reporting Agency
    80 A1          Type of this line ("1"), can be blank if first
    -              line of event

    If more than 3 magnitudes need to be associated with the hypocenter in the
    first line, a subsequent additional type one line can be written with the
    same year, month, day until event ID and hypocenter agency. The magnitudes
    on this line will then be associated with the main header line and there
    is then room for 6 magnitudes.
    """

    def __init__(self):
        self.year = None
        self.month = None
        self.day = None
        self.fixed_origin_time = ' '
        self.hour = None
        self.minutes = None
        self.seconds = None
        self.location_model_indicator = ' '
        self.distance_indicator = ' '
        self.event_id = ' '
        self.latitude = None
        self.longitude = None
        self.depth = None
        self.depth_indicator = ' '
        self.locating_indicator = ' '
        self.hypocenter_reporting_agency = ' '
        self.number_of_stations_used = None
        self.rms_of_time_residuals = None
        self.magnitude_1 = None
        self.type_of_magnitude_1 = ' '
        self.magnitude_reporting_agency_1 = ' '
        self.magnitude_2 = None
        self.type_of_magnitude_2 = ' '
        self.magnitude_reporting_agency_2 = ' '
        self.magnitude_3 = None
        self.type_of_magnitude_3 = ' '
        self.magnitude_reporting_agency_3 = ' '
        self.dataid = ''


class seisan_2():
    """
    Type 2 line (Macroseismic information).

    Cols Format Description
    1-5           Blank
    6-20          a Any descriptive text
    21            Free
    22    a1      Diastrophism code (PDE type)
    -                    F = Surface faulting
    -                    U = Uplift or subsidence
    -                    D = Faulting and Uplift/Subsidence
    23    a1      Tsunami code (PDE type)
    -                    T = Tsunami generated
    -                    Q = Possible tsunami
    24    a1      Seiche code (PDE type)
    -                    S = Seiche
    -                    Q = Possible seiche
    25    a1      Cultural effects (PDE type)
    -                    C = Casualties reported
    -                    D = Damage reported
    -                    F = Earthquake was felt
    -                    H = Earthquake was heard
    26    a1      Unusual events (PDE type)
    -                    L = Liquefaction
    -                    G = Geyser/fumarole
    -                    S = Landslides/Avalanches
    -                    B = Sand blows
    -                    C = Cracking in the ground (not normal faulting).
    -                    V = Visual phenomena
    -                    O = Olfactory phenomena
    -                    M = More than one of the above observed.
    27            Free
    28-29 i2      Max Intensity
    30    a1      Max Intensity qualifier
    -                    (+ or - indicating more precisely the intensity)
    31-32 a2      Intensity scale (ISC type definitions)
    -                    MM = Modified Mercalli
    -                    RF = Rossi Forel
    -                    CS = Mercalli - Cancani - Seberg
    -                    SK = Medevev - Sponheur - Karnik
    33            Free
    34-39 f6.2    Macroseismic latitude (Decimal)
    40            Free
    41-47 f7.2    Macroseismic longitude (Decimal)
    48            Free
    49-51 f3.1    Macroseismic magnitude
    52    a1      Type of magnitude
    -                    I = Magnitude based on maximum Intensity.
    -                    A = Magnitude based on felt area.
    -                    R = Magnitude based on radius of felt area.
    -                    * = Magnitude calculated by use of special formulas
    -                        developed by some person for a certain area.
    -                        Further info should be given on line 3.
    53-56 f4.2    Logarithm (base 10) of radius of felt area.
    57-61 f5.2    Logarithm (base 10) of area (km**2) number 1 where
    -                    earthquake was felt exceeding a given intensity.
    62-63 i2      Intensity bordering the area number 1.
    64-68 f5.2    Logarithm (base 10) of area (km**2) number 2 where
    -                    earthquake was felt exceeding a given intensity.
    69-70 i2      Intensity bordering the area number 2.
    71            Free
    72    a1      Quality rank of the report (A, B, C, D)
    73-75 a3      Reporting agency
    76-79         Free
    80    a1      Type of this line ("2")
    """

    def __init__(self):
        self.description = ' '
        self.diastrophism_code = ' '
        self.tsunami_code = ' '
        self.seiche_code = ' '
        self.cultural_effects = ' '
        self.unusual_events = ' '
        self.max_intensity = None
        self.max_intensity_qualifier = ' '
        self.intensity_scale = ' '
        self.macroseismic_latitude = None
        self.macroseismic_longitude = None
        self.macroseismic_magnitude = None
        self.type_of_magnitude = ' '
        self.log_of_felt_area_radius = None
        self.log_of_area_1 = None
        self.intensity_bordering_area_1 = None
        self.log_of_area_2 = None
        self.intensity_bordering_area_2 = None
        self.quality_rank = ' '
        self.reporting_agency = ' '
        self.dataid = ''


class seisan_3():
    """
    Type 3 Line (Optional).

    Columns Format Description 	Comments
    1               Free
    2-79    A       Text      	Anything
    80      A1      Type of this line ("3")
    """

    def __init__(self):
        self.text = ''
        self.region = ''
        self.dataid = ''


class seisan_4():
    """
    Type 4 line.

    Columns Format Description 	Comments

    1               Free
    2- 6    A5      Station Name    Blank = End of readings = end of event
    7       A1      Instrument Type S = SP, I = IP, L = LP etc
    8       A1      Component 	   Z, N, E ,T, R, 1, 2
    9               Free or weight, see note below
    10      A1      Quality Indicator 	I, E, etc.
    11-14   A2      Phase ID 	PN, PG, LG, P, S, etc. **
    15      I1      Weighting Indicator (1-4) 0 or blank= full weight, 1=75%,
    -               2=50%, 3=25%, 4=0%, 9: no weight, use difference time
    -               (e.g. P-S).
    16              Free or flag A to indicate automatic pick, removed when
    -               picking
    17      A1      First Motion    C, D
    18              Note: Currently 15 to 18 can also be used for phase
    -               assuming column 11-14 is not blank. See note ** below.
    19-20   I2      Hour 	Hour can be up to 48 to indicate next day
    21-22   I2      Minutes
    23-28   F6.0    Seconds
    29              Free
    30-33   I4      Duration (to noise) 	Seconds
    34-40   g7.1    Amplitude (Zero-Peak) in units of nm, nm/s, nm/s^2 or
    -               counts.
    41              Free
    42-45   F4.0    Period 	Seconds
    46              Free
    47-51   F5.0    Direction of Approach 	Degrees
    52              Free
    53-56   F4.0    Phase Velocity 	Km/second
    57-60   F4.0    Angle of incidence (was Signal to noise ratio before
    -               version 8)
    61-63   I3      Azimuth residual
    64-68   F5.1    Travel time residual
    69-70   I2      Weight
    71-75   F5.0    Epicentral distance(km)
    76              Free
    77-79   I3      Azimuth at source
    80      A1      Type of this line ("4"), can be blank, which it is most
    -               often.

    NB: Epicentral distance: Had format I5 before version 7.2. All old lines
    can be read with format F5.0 with same results, but now distance can also
    be e.g. 1.23 km which cannot be read by earlier versions. However, an
    UPDATE would fix that.
    ** Long phase names: An 8 character phase can be used in column 11-18.
    There is then not room for polarity information. The weight is then put
    into column 9. This format is recognized by HYP and MULPLT.

    Type 4 cards should be followed by a Blank Card (Type 0)
    """

    def __init__(self):
        self.station_name = ' '
        self.instrument_type = ' '
        self.component = ' '
        self.quality = ' '
        self.phase_id = None
        self.weighting_indicator = None
        self.flag_auto_pick = None
        self.first_motion = ' '
        self.hour = None
        self.minutes = None
        self.seconds = None
        self.duration = None
        self.amplitude = None
        self.period = None
        self.direction_of_approach = None
        self.phase_velocity = None
        self.angle_of_incidence = None
        self.azimuth_residual = None
        self.travel_time_residual = None
        self.weight = None
        self.epicentral_distance = None
        self.azimuth_at_source = None
        self.dataid = ''


class seisan_5():
    """
    Type 5 line (optional).

    Error estimates of previous line, currently not used by any SEISAN
    programs.

    Columns Format Description 	Comments
    1       Free
    2-79    Error estimates in same format as previous line, normally type 4
    80 A1   Type of this line ("5")
    """

    def __init__(self):
        self.text = ' '
        self.dataid = ''


class seisan_6():
    """
    Type 6 Line (Optional).

    Columns Format Description 	Comments
    1       Free
    2-79    A      Name(s) of tracedata files
    80      A1     Type of this line ("6")
    """

    def __init__(self):
        self.tracedata_files = ' '
        self.dataid = ''


class seisan_7():
    """
    Type 7 Line (Optional).

    Columns Format Description 	Comments
    1       Free
    2-79    A      Help lines to place the numbers in right positions
    80      A1     Type of this line ("7")

    STAT SP : Station and component
    IPHAS : Phase with onset
    W : Phase weight, HYPO71 style
    HRMM SECON : Hour, minute and seconds
    CODA : Coda length (secs)
    AMPLIT PERI: Amplitude (nm) and period (sec)
    AZIM VELO : Back azimuth (deg) and apparent velocity of arrival at station
    AIN : Angle of incidence
    AR : back azimuth residual
    TRES : Arrival time residual
    W : Weigh used in location
    DIS : Epicentral distance in km
    CAZ : Azimuth from event to station
    """

    def __init__(self):
        self.stat = None
        self.sp = None
        self.iphas = None
        self.phase_weight = None
        self.d = None
        self.hour = None
        self.minutes = None
        self.seconds = None
        self.coda = None
        self.amplitude = None
        self.period = None
        self.azimuth = None
        self.velocity = None
        self.angle_incidence = None
        self.azimuth_residual = None
        self.time_residual = None
        self.location_weight = None
        self.distance = None
        self.caz = None
        self.dataid = ''


class seisan_E():
    """
    Type E Line (Optional): Hyp error estimates.

    Columns Format Description

    1     Free
    2-5   A4 The text GAP=
    6-8   I3 Gap
    15-20 F6.2 Origin time error
    25-30 F6.1 Latitude (y) error
    31-32 Free
    33-38 F6.1 Longitude (x) error (km)
    39-43 F5.1 Depth (z) error (km)
    44-55 E12.4 Covariance (x,y) km*km
    56-67 E12.4 Covariance (x,z) km*km
    68-79 E14.4 Covariance (y,z) km*km

    covariance matrix:
    var(1,1)=erx*erx
    var(2,2)=ery*ery
    var(3,3)=erz*erz
    var(1,2)=cvxy
    var(1,3)=cvxz
    var(2,3)=cvyz
    var(2,1)=var(1,2)
    var(3,1)=var(1,3)
    var(3,2)=var(2,3)
    """

    def __init__(self):
        self.gap = None
        self.origin_time_error = None
        self.latitude_error = None  # ery
        self.longitude_error = None  # erx
        self.depth_error = None  # erz
        self.cov_xy = None  # cvxy
        self.cov_xz = None  # cvxz
        self.cov_yz = None  # cvyz
        self.dataid = ''


class seisan_F():
    """
    Type F Line (Optional): Fault plane solution.

    Columns Format Description

    1:30    3F10.0 Strike, dip and rake, Aki convention
    31:45   4F5.1  Error in strike dip and rake (HASH), error in fault plane
    -              and aux. plane (FPFIT)
    46:50   F5.1   Fit error:  FPFIT and HASH (F-fit)
    51:55   F5.1   Station distribution ratio (FPFIT, HASH)
    56:60   F5.1   Amplitude ratio fit (HASH, FOCMEC)
    61:65   I2     Number of bad polarities (FOCMEC, PINV)
    64.65   I2     Number of bad amplitude  ratios (FOCMEC)
    67:69   A3     Agency code
    71:77   A7     Program used
    78:78   A1     Quality of solution, A (best), B C or D (worst), added
    -              manually
    79:79   A1     Blank, can be used by user
    80:80   A1     F
    """

    def __init__(self):
        self.strike = None
        self.dip = None
        self.rake = None
        self.err1 = None
        self.err2 = None
        self.err3 = None
        self.fit_error = None
        self.station_distribution_ratio = None
        self.amplitude_ratio = None
        self.number_of_bad_polarities = None
        self.number_of_bad_amplitude_ratios = None
        self.agency_code = ' '
        self.program_used = ' '
        self.solution_quality = ' '
        self.dataid = ''


class seisan_H():
    """
    Type H line, High accuracy hypocenter line.

    Columns Format Description
    1:15    As type 1 line
    16      Free
    17      f6.3    Seconds
    23      Free
    24:32   f9.5    Latitude
    33      Free
    34:43   f10.5   Longitude
    44      Free
    45:52   f8.3    Depth
    53      Free
    54:59   f6.3    RMS
    60:79   Free
    80      H
    """

    def __init__(self):
        self.year = None
        self.month = None
        self.day = None
        self.fixed_origin_time = ' '
        self.hour = None
        self.minutes = None
        self.seconds = None
        self.latitude = None
        self.longitude = None
        self.depth = None
        self.rms = None
        self.dataid = ''


class seisan_I():
    """
    Type I Line, ID line.

    Columns Description
    1       Free
    2:8     Help text for the action indicator
    9:11 	Last action done, so far defined:
    -        SPL: Split
    -        REG: Register
    -        ARG: AUTO Register, AUTOREG
    -        UPD: Update
    -        UP : Update only from EEV
    -        REE: Register from EEV
    -        DUB: Duplicated event
    -        NEW: New event
    12      Free
    13:26   Date and time of last action
    27      Free
    28:30   Help text for operator
    36:42   Help text for status
    43:56   Status flags, not yet defined
    57      Free
    58:60   Help text for ID
    61:74   ID, year to second
    75      If d, this indicate that a new file id had to be created which was
    -       one or more seconds different from an existing ID to avoid
    -       overwrite.
    76      Indicate if ID is locked. Blank means not locked, L means locked.
    """

    def __init__(self):
        self.last_action_done = ' '
        self.date_time_of_last_action = ' '
        self.operator = ' '
        self.status = ' '
        self.id = ' '
        self.new_id_created = ' '
        self.id_locked = ' '
        self.dataid = ''
# This is used in a type 3 line
        self.region = ''


class seisan_M():
    """
    Type M Line (Optional): Moment tensor solution.

    Note: the type M lines are pairs of lines with one line that gives the
    hypocenter time, and one line that gives the moment tensor values:

    The first moment tensor line:
    Columns Format Description
    1:1            Free
    2:5     I4     Year
    7:8     I2     Month
    9:10    I2     Day of Month
    12:13   I2     Hour
    14:15   I2     Minutes
    17:20   F4.1   Seconds
    24:30   F7.3   Latitude                       Degrees (+ N)
    31:38   F8.3   Longitude                      Degrees (+ E)
    39:43   F5.1   Depth                          Km
    46:48   A3     Reporting Agency
    56:59   F4.1   Magnitude
    60      A1     Type of Magnitude L=ML, b=mb, B=mB, s=Ms, S=MS, W=MW,
    61:63   A3     Magnitude Reporting Agency
    71:77   A7     Method used
    78:78   A1     Quality of solution, A (best), B C or D (worst), added
    -              manually
    79:79   A1     Blank, can be used by user
    80:80   A1     M

    The second moment tensor line:
    Columns Format Description
    1:1            Free
    2:3     A2     MT
    4:9     F6.3   Mrr or Mzz [Nm]
    11:16   F6.3   Mtt or Mxx [Nm]
    18:23   F6.3   Mpp or Myy [Nm]
    25:30   F6.3   Mrt or Mzx [Nm]
    32:37   F6.3   Mrp or Mzy [Nm]
    39:44   F6.3   Mtp or Mxy [Nm]
    46:48   A3     Reporting Agency
    49:49   A1     MT coordinate system (S=spherical, C=Cartesian)
    50:51   i2     Exponential
    53:62   G6.3   Scalar Moment [Nm]
    71:77   A7     Method used
    78:78   A1     Quality of solution, A (best), B C or D (worst), added
    -              manually
    79:79   A1     Blank, can be used by user
    80:80   A1     M
    """

    def __init__(self):
        self.year = None
        self.month = None
        self.day = None
        self.hour = None
        self.minutes = None
        self.seconds = None
        self.latitude = None
        self.longitude = None
        self.depth = None
        self.reporting_agency = ' '
        self.megnitude = None
        self.magnitude_type = ' '
        self.magnitude_reporting_agency = ' '
        self.method_used = ' '
        self.quality = ' '
        self.mt = ' '
        self.mrr_mzz = None
        self.mtt_mxx = None
        self.mpp_myy = None
        self.mrt_mzx = None
        self.mrp_mzy = None
        self.mtp_mxy = None
        self.reporting_agency2 = ' '
        self.mt_coordinate_system = ' '
        self.exponential = None
        self.scalar_moment = None
        self.method_used_2 = ' '
        self.quality_2 = ' '
        self.dataid = ''


class seisan_P():
    """
    Type P line, file name of a picture file.

    Columns Description
    1:1      Free
    2:79     File name
    80:80    P
    """

    def __init__(self):
        self.filename = ' '
        self.dataid = ''
