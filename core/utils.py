"""Some general utilities that are useful."""

import time

import scipy as sp
from scipy.interpolate import interp1d
from scipy import linalg
import ephem

def elaz2radec_lst(el, az, lst, lat = 38.43312) :
    """DO NOT USE THIS ROUTINE FOR ANTHING THAT NEEDS TO BE RIGHT.  IT DOES NOT
    CORRECT FOR PRECESSION.

    Calculates the Ra and Dec from elavation, aximuth, LST and Latitude.

    This function is vectorized with numpy so should be fast.  Standart numpy
    broadcasting should also work.

    All angles in degrees, lst in seconds. Latitude defaults to GBT.
    """

    # Convert everything to radians.
    el = sp.radians(el)
    az = sp.radians(az)
    lst = sp.array(lst, dtype = float)*2*sp.pi/86400
    lat = sp.radians(lat)
    # Calculate dec.
    dec = sp.arcsin(sp.sin(el)*sp.sin(lat) +
                    sp.cos(el)*sp.cos(lat)*sp.cos(az))
    # Calculate the hour angle
    ha = sp.arccos((sp.sin(el) - sp.sin(lat)*sp.sin(dec)) /
                   (sp.cos(lat)*sp.cos(dec)))
    ra = sp.degrees(lst - ha) % 360

    return ra, sp.degrees(dec)

def elaz2radecGBT(el, az, UT) :
    """Calculates the Ra and Dec from the elevation, azimuth and UT for an
    observer at GBT.

    All input should be formated to correspond to the data in a GBT fits file.
    El, and Az in degrees and UT a string like in the GBT DATE-OBS field.

    Largely copied from Kevin's code.
    """

    GBT = ephem.Observer()
    GBT.long = '-79:50:23.4'
    GBT.lat = '38:25:59.23'
    GBT.pressure = 0 # no refraction correction.
    GBT.temp = 0

    UT_wholesec, partial_sec = UT.split('.', 1)
    time_obj = time.strptime(UT_wholesec, "%Y-%m-%dT%H:%M:%S")
    UT_reformated = time.strftime("%Y/%m/%d %H:%M:%S", time_obj)
    GBT.date = UT_reformated + "." + partial_sec

    el_r = el*sp.pi/180.0
    az_r = az*sp.pi/180.0
    ra, dec = GBT.radec_of(az_r,el_r)

    return ra*180.0/sp.pi, dec*180.0/sp.pi

def LSTatGBT(UT) :
    """Calculates the LST from the UT of an observer at GBT.

    All input should be formated to correspond to the data in a GBT fits file.
    UT a string like in the GBT DATE-OBS field.

    Largely copied from Kevin's code.
    """

    GBT = ephem.Observer()
    GBT.long = '-79:50:23.4'
    GBT.lat = '38:25:59.23'
    GBT.pressure = 0 # no refraction correction.
    GBT.temp = 0

    UT_wholesec, partial_sec = UT.split('.', 1)
    time_obj = time.strptime(UT_wholesec, "%Y-%m-%dT%H:%M:%S")
    UT_reformated = time.strftime("%Y/%m/%d %H:%M:%S", time_obj)
    GBT.date = UT_reformated + "." + partial_sec

    LST = GBT.sidereal_time() #IN format xx:xx:xx.xx ?

    return LST*180.0/sp.pi

def time2float(UT) :
    """Calculates float seconds from a time string.

    Convert a time string in format %Y-%m-%dT%H:%M:%S.partial to a float number
    of seconds ignaroing all posible corrections."""

    UT_wholesec, partial_sec = UT.split('.', 1)
    to = time.strptime(UT_wholesec, "%Y-%m-%dT%H:%M:%S")
    return (float('0.' + partial_sec) + to.tm_sec + 60*(to.tm_min +
            60*(to.tm_hour + 24*(to.tm_yday + 365*(to.tm_year-2000)))))

def mk_map_grid(centre, shape, spacing) :
    """Make a grid of coordinates in Ra and Dec.

    This function accepts a field centre (tuple 2 floats), map-shape (tuple 2
    ints) and pixel spacing (float) for a map.  It returns two arrays with
    the specified map shape, and meshed like sp.meshgrid.  However, the spacing
    in Ra is converted to real degrees (devided by cos(dec) at field centre).

    All units in degrees, values are pixel centres.
    """

    dec = centre[1] + spacing*sp.arange(-(shape[1]-1.)/2., shape[1]/2.)
    ra = centre[0] + (spacing/sp.cos(centre[1]*sp.pi/180.) *
                      sp.arange(-(shape[0]-1.)/2., shape[0]/2.))

    grid_ra, grid_dec = sp.meshgrid(ra, dec)

    return grid_ra, grid_dec

def get_beam(freq) :
    """Get the GBT beam width at a frequency (or an array of frequencies).

    This is currently pretty rough and only uses on scans worth of data.
    """

    # This data is pretty rough.  Just cut and pasted from one scan, not
    # averaged.
    beam_data = [0.316148488246, 0.306805630985, 0.293729620792,
                 0.281176247549, 0.270856788455, 0.26745856078,
                 0.258910010848, 0.249188429031]
    freq_data = sp.array([695, 725, 755, 785, 815, 845, 875, 905], dtype=float)
    freq_data *= 1.0e6
    f = interp1d(freq_data, beam_data, bounds_error=False, fill_value = -1)
    b = f(freq)
    b[b<0] = 0.316148488246
    return b

def polint2str(pol_int) :
    """Convert an interger representing a polarization to a representing the
    polarization.

    This is based on the SDfits convention that I pulled from: 
        https://safe.nrao.edu/wiki/bin/view/Main/SdfitsDetails

    Here are the return values based on the passed integer.

    RR  -1
    LL 	-2
    RL 	-3
    LR 	-4
    XX 	-5
    YY 	-6
    XY 	-7
    YX 	-8
    I 	1
    Q 	2
    U 	3
    V 	4
    Otherwise raises a ValueError.
    """

    if pol_int == -1 :
        return 'RR'
    elif pol_int == -2 :
        return 'LL'
    elif pol_int == -3 :
        return 'RL'
    elif pol_int == -4 :
        return 'LR'
    elif pol_int == -5 :
        return 'XX'
    elif pol_int == -6 :
        return 'YY'
    elif pol_int == -7 :
        return 'XY'
    elif pol_int == -8 :
        return 'YX'
    elif pol_int == 1 :
        return 'I'
    elif pol_int == 2 :
        return 'Q'
    elif pol_int == 3 :
        return 'U'
    elif pol_int == 4 :
        return 'V'
    else :
        raise ValueError("Polarization integer must be in range(-8, 5) and "
                         "nonzero")

def ampfit(data, covariance, theory):
    """Fits the amplitude of the theory curve to the data.

    Finds `amp` such that `amp`*`theory` is the best fit to `data`.

    Returns
    -------
    amp : float
        Fitted amplitude.
    errir : float
        Error on fitted amplitude.
    """

    data = sp.asarray(data)
    covariance = sp.asarray(covariance)
    theory = sp.asarray(theory)
    # Sanity check inputs.
    if len(data.shape) != 1:
        raise ValueError("`data` must be a 1D vector.")
    n = len(data)
    if data.shape != theory.shape:
        raise ValueError("`theory` must be the same shape as `data`.")
    if covariance.shape != (n,n):
        msg = "`covariance` must be a square matrix compatible with data."
        raise ValueError(msg)
    # Linear fit for the amplitude.  Formulas July 24, 2011 of Kiyo's notebook.
    covariance_inverse = linalg.inv(covariance)
    weighted_data = sp.dot(covariance_inverse, data)
    amp = sp.dot(theory, weighted_data)
    normalization = sp.dot(covariance_inverse, theory)
    normalization = sp.dot(theory, normalization)
    amp /= normalization
    # Calculate the Error.
    error = sp.sqrt(1/normalization)

    return amp, error

