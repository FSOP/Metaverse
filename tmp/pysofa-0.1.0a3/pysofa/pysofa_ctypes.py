# -*- coding: utf-8 -*-

#
# Copyright 2010 Frédéric Grollier
#
# Distributed under the terms of the MIT license
#

from ctypes import CDLL, byref, POINTER
from ctypes.util import find_library
from ctypes import c_int, c_char, c_double

from numpy.ctypeslib import ndpointer
from numpy import ndarray, array, zeros, asarray, asfarray, asmatrix

sofalib_filename = find_library('sofa_c')
if sofalib_filename is None:
    raise ImportError('Unable to find the shared C library "sofa_c".')
sofa = CDLL(sofalib_filename)



# iauA2af
sofa.iauA2af.argtypes = [c_int, #ndp
                            c_double, #angle
                            POINTER(c_char), #sign
                            c_int * 4] #idmsf
def A2af(ndp, angle):
    """ Decompose radians into degrees, arcminutes, arcseconds, fraction.

    :param ndp: the requested resolution.
    :type ndp: int

    :param angle: the value to decompose.
    :type angle: float

    :returns: a tuple whose first member is a string containing the sign, and
        the second member is itself a tuple (degrees, arcminutes, arcseconds,
        fraction).

    *sofa manual.pdf page 19*
    """
    sign = c_char()
    idmsf = (c_int * 4)()
    sofa.iauA2af(ndp, float(angle), byref(sign), idmsf)
    return sign.value, tuple([v for v in idmsf])


# iauA2tf
sofa.iauA2tf.argtypes = [c_int, #ndp
                            c_double, #angle
                            POINTER(c_char), #sign
                            c_int * 4] #ihmsf
def A2tf(ndp, angle):
    """ Decompose radians into hours, arcminutes, arcseconds, fraction.

    :param ndp: the requested resolution.
    :type ndp: int

    :param angle: the value to decompose.
    :type angle: float

    :returns: a tuple whose first member is a string containing the sign, and
        the second member is itself a tuple (hours, arcminutes, arcseconds,
        fraction).

    *sofa manual.pdf page 20*
    """
    sign = c_char()
    ihmsf = (c_int * 4)()
    sofa.iauA2tf(ndp, float(angle), byref(sign), ihmsf)
    return sign.value, tuple([v for v in ihmsf])


# iauAf2a
sofa.iauAf2a.argtypes = [c_char, #sign
                            c_int, #ideg
                            c_int, #iamin
                            c_double, #asec
                            POINTER(c_double)] #rad
sofa.iauAf2a.restype = c_int
def Af2a(s, ideg, iamin, asec):
    """ Convert degrees, arcminutes, arcseconds to radians.

    :param s: sign, '-' for negative, otherwise positive.

    :param ideg: degrees.
    :type ideg: int

    :param iamin: arcminutes.
    :type iamin: int

    :param asec: arcseconds.
    :type asec: float

    :returns: a 2-tuple

        * the converted value in radians as a float
        * function status.

    *sofa manual.pdf page 21*
    """

    rad = c_double()
    s = sofa.iauAf2a(str(s), ideg, iamin, asec, byref(rad))
    return rad.value, s



# iauAnp
sofa.iauAnp.argtypes = [c_double]
sofa.iauAnp.restype = c_double
def Anp(a):
    """ Normalize *a* into the range 0 <= result < 2pi.

    :param a: the value to normalize.
    :type a: float

    :returns: the normalized value as a float.

    *sofa manual.pdf page 22*
    """
    return sofa.iauAnp(float(a))


# iauAnpm
sofa.iauAnpm.argtypes = [c_double]
sofa.iauAnpm.restype = c_double
def Anpm(a):
    """ Normalize *a* into the range -pi <= result < +pi.

    :param a: the value to normalize.
    :type a: float

    :returns: the normalized value as a float.

    *sofa manual.pdf page 23*
    """
    return sofa.iauAnpm(float(a))


# iauBi00
sofa.iauBi00.argtypes = [POINTER(c_double), #dpsibi
                            POINTER(c_double), #depsbi
                            POINTER(c_double)] #dra
def Bi00():
    """ Frame bias components of IAU 2000 precession-nutation models.

    :returns: a tuple of three items:

        * longitude correction (float)
        * obliquity correction (float)
        * the ICRS RA of the J2000.0 mean equinox (float).

    *sofa manual.pdf page 24*
    """
    dpsibi = c_double()
    depsbi = c_double()
    dra = c_double()
    sofa.iauBi00(byref(dpsibi), byref(depsbi), byref(dra))
    return dpsibi.value, depsbi.value, dra.value

# iauBp00
sofa.iauBp00.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float), #rb
                            ndpointer(shape=(3,3), dtype=float), #rp
                            ndpointer(shape=(3,3), dtype=float)] #rbp
def Bp00(date1, date2):
    """ Frame bias and precession, IAU 2000.

    :param date1, date2: TT as a two-part Julian date.

    :returns: a tuple of three items:

        * frame bias matrix (numpy.matrix of shape 3x3)
        * precession matrix (numpy.matrix of shape 3x3)
        * bias-precession matrix (numpy.matrix of shape 3x3)

    *sofa manual.pdf page 25*
    """
    rb = asmatrix(zeros(shape=(3,3), dtype=float))
    rp = asmatrix(zeros(shape=(3,3), dtype=float))
    rbp = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauBp00(date1, date2, rb, rp, rbp)
    return rb, rp, rbp


# iauBp06
sofa.iauBp06.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float), #rb
                            ndpointer(shape=(3,3), dtype=float), #rp
                            ndpointer(shape=(3,3), dtype=float)] #rbp
def Bp06(date1, date2):
    """ Frame bias and precession, IAU 2006.

    :param date1, date2: TT as a two-part Julian date.

    :returns: a tuple of three items:

        * frame bias matrix (numpy.matrix of shape 3x3)
        * precession matrix (numpy.matrix of shape 3x3)
        * bias-precession matrix (numpy.matrix of shape 3x3)

    *sofa manual.pdf page 27*
    """
    rb = asmatrix(zeros(shape=(3,3), dtype=float))
    rp = asmatrix(zeros(shape=(3,3), dtype=float))
    rbp = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauBp06(date1, date2, rb, rp, rbp)
    return rb, rp, rbp


# iauBpn2xy
sofa.iauBpn2xy.argtypes = [ndpointer(shape=(3,3), dtype=float), #rbpn
                            POINTER(c_double), #x
                            POINTER(c_double)] #y
def Bpn2xy(rbpn):
    """ Extract from the bias-precession-nutation matrix the X,Y coordinates
    of the Celestial Intermediate Pole.

    :param rbpn: celestial-to-true matrix
    :type rbpn: numpy.ndarray, matrix or nested sequences of shape 3x3

    :returns: a tuple of two items containing *x* and *y*, as floats.

    *sofa manual.pdf page 28*
    """
    x = c_double()
    y = c_double()
    sofa.iauBpn2xy(asmatrix(rbpn, dtype=float), x, y)
    return x.value, y.value


# iauC2i00a
sofa.iauC2i00a.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rc2i
def C2i00a(date1, date2):
    """ Form the celestial-to-intermediate matrix for a given date using the
    IAU 2000A precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.

    :returns: the celestial-to-intermediate matrix, as a numpy.matrix of
        shape 3x3.

    *sofa manual.pdf page 29*
    """
    rc2i = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2i00a(date1, date2, rc2i)
    return rc2i


# iauC2i00b
sofa.iauC2i00b.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rc2i
def C2i00b(date1, date2):
    """ Form the celestial-to-intermediate matrix for a given date using the
    IAU 2000B precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.

    :returns: the celestial-to-intermediate matrix, as a numpy.matrix of
        shape 3x3.

    *sofa manual.pdf page 31*
    """
    rc2i = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2i00b(date1, date2, rc2i)
    return rc2i


# iauC2i06a
sofa.iauC2i06a.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rc2i
def C2i06a(date1, date2):
    """ Form the celestial-to-intermediate matrix for a given date using the
    IAU 2006 precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.

    :returns: the celestial-to-intermediate matrix, as a numpy.matrix of
        shape 3x3.

    *sofa manual.pdf page 33*
    """
    rc2i = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2i06a(date1, date2, rc2i)
    return rc2i


# iauC2ibpn
sofa.iauC2ibpn.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float), #rbpn
                            ndpointer(shape=(3,3), dtype=float)] #rc2i
def C2ibpn(date1, date2, rbpn):
    """ Form the celestial-to-intermediate matrix for a given date given the
    bias-precession-nutation matrix. IAU 2000.

    :param date1, date2: TT as a two-part Julian date.

    :param rbpn: celestial-to-true matrix.
    :type rbpn: numpy.ndarray, numpy.matrix or nested sequences of shape 3x3

    :returns: the celestial-to-intermediate matrix, as a numpy.matrix of
        shape 3x3.

    *sofa manual.pdf page 34*
    """
    rc2i = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2ibpn(date1, date2, asmatrix(rbpn, dtype=float), rc2i)
    return rc2i


# iauC2ixy
sofa.iauC2ixy.argtypes = [c_double, #date1
                            c_double, #date2
                            c_double, #x
                            c_double, #y
                            ndpointer(shape=(3,3), dtype=float)] #rc2i
def C2ixy(date1, date2, x, y):
    """ Form the celestial to intermediate-frame-of-date matrix for a given
    date when CIP X,Y coordinates are known. IAU 2000.

    :param date1, date2: TT as a two-part Julian date.

    :param x, y: celestial intermediate pole coordinates.
    :type x, y: float

    :returns: the celestial-to-intermediate matrix as a numpy.matrix of shape
        3x3.

    *sofa manual.pdf page 36*
    """
    rc2i = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2ixy(date1, date2, float(x), float(y), rc2i)
    return rc2i


# iauC2ixys
sofa.iauC2ixys.argtypes = [c_double, #x
                            c_double, #y
                            c_double, #s
                            ndpointer(shape=(3,3), dtype=float)] #rc2i
def C2ixys(x, y, s):
    """ Form the celestial to intermediate-frame-of-date matrix given the CIP
    X,Y coordinates and the CIO locator s.

    :param x, y: celestial intermediate pole coordinates.
    :type x, y: float

    :param s: the CIO locator.
    :type s: float

    :returns: the celestial-to-intermediate matrix as a numpy.matrix of shape
        3x3.

    *sofa manual.pdf page 38*
    """
    rc2i = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2ixys(float(x), float(y), float(s), rc2i)
    return rc2i


# iauC2s
sofa.iauC2s.argtypes = [ndpointer(shape=(1,3), dtype=float), #p
                        POINTER(c_double), #theta
                        POINTER(c_double)] #phi
def C2s(p):
    """ P-vector to spherical coordinates.

    :param p: p-vector
    :type p: numpy.ndarray, matrix or nested sequences of shape (1,3)

    :returns: a tuple of two items:

        * the longitude angle in radians (float)
        * the latitude angle in radians (float)

    *sofa manual.pdf page 39*
    """
    theta = c_double()
    phi = c_double()
    sofa.iauC2s(asmatrix(p, dtype=float), byref(theta), byref(phi))
    return theta.value, phi.value


# iauC2t00a
sofa.iauC2t00a.argtypes = [c_double, #tta
                            c_double, #ttb
                            c_double, #uta
                            c_double, #utb
                            c_double, #xp
                            c_double, #yp
                            ndpointer(shape=(3,3), dtype=float)]
def C2t00a(tta, ttb, uta, utb, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1 and
    the polar motion, using IAU 2000A nutation model.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: the celestial-to-terrestrial matrix, as a numpy.matrix of shape
        3x3.

    *sofa manual.pdf page 40*
    """
    rc2t = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2t00a(tta, ttb, uta, utb, float(xp), float(yp), rc2t)
    return rc2t


# iauC2t00b
sofa.iauC2t00b.argtypes = [c_double, #tta
                            c_double, #ttb
                            c_double, #uta
                            c_double, #utb
                            c_double, #xp
                            c_double, #yp
                            ndpointer(shape=(3,3), dtype=float)]
def C2t00b(tta, ttb, uta, utb, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1 and
    the polar motion, using IAU 2000B nutation model.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: the celestial-to-terrestrial matrix, as a numpy.matrix of shape
        3x3.

    *sofa manual.pdf page 42*
    """
    rc2t = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2t00b(tta, ttb, uta, utb, float(xp), float(yp), rc2t)
    return rc2t


# iauC2t06a
sofa.iauC2t06a.argtypes = [c_double, #tta
                            c_double, #ttb
                            c_double, #uta
                            c_double, #utb
                            c_double, #xp
                            c_double, #yp
                            ndpointer(shape=(3,3), dtype=float)]
def C2t06a(tta, ttb, uta, utb, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1 and
    the polar motion, using the IAU 2006 precession and IAU 2000A nutation
    models.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: the celestial-to-terrestrial matrix, as a nunmp.matrix of shape
        3x3.

    *sofa manual.pdf page 44*
    """
    rc2t = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2t06a(tta, ttb, uta, utb, float(xp), float(yp), rc2t)
    return rc2t


# iauC2tcio
sofa.iauC2tcio.argtypes = [ndpointer(shape=(3,3), dtype=float), #rc2i
                            c_double, #era
                            ndpointer(shape=(3,3), dtype=float), #rpom
                            ndpointer(shape=(3,3), dtype=float)] #rc2t
def C2tcio(rc2i, era, rpom):
    """ Assemble the celestial-to-terrestrial matrix from CIO-based
    components (the celestial-to-intermediate matrix, the Earth Rotation Angle
    and the polar motion matrix).

    :param rc2i: celestial-to-intermediate matrix.
    :type rc2i: array-like object of shape (3,3)

    :param era: Earth rotation angle
    :type era: float

    :param rpom: polar-motion matrix.
    :type rpom: array-like of shape (3,3)

    :returns: celestial-to-terrestrial matrix as a numpy.matrix of shape
        3x3.

    *sofa manual.pdf page 46*
    """
    rc2t = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2tcio(asmatrix(rc2i, dtype=float), float(era),
                                            asmatrix(rpom, dtype=float), rc2t)
    return rc2t


# iauC2teqx
sofa.iauC2teqx.argtypes = [ndpointer(shape=(3,3), dtype=float), #rbpn
                            c_double, #gst
                            ndpointer(shape=(3,3), dtype=float), #rpom
                            ndpointer(shape=(3,3), dtype=float)] #rc2t
def C2teqx(rbpn, gst, rpom):
    """ Assemble the celestial-to-terrestrial matrix from equinox-based
    components (the celestial-to-true matrix, the Greenwich Apparent Sidereal
    Time and the polar motion matrix).

    :param rbpn: celestial-to-true matrix.
    :type rbpn: array-like of shape (3,3)

    :param gst: Greenwich apparent sidereal time.
    :type gst: float

    :param rpom: polar-motion matrix.
    :type rpom: array-like of shape (3,3)

    :returns: celestial-to-terrestrial matrix as a numpy.matrix of shape
        3x3.

    *sofa manual.pdp page 47*
    """
    rc2t = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2teqx(asmatrix(rbpn, dtype=float), float(gst),
                                            asmatrix(rpom, dtype=float), rc2t)
    return rc2t


# iauC2tpe
sofa.iauC2tpe.argtypes = [c_double, #tta,
                            c_double, #ttb
                            c_double, #uta
                            c_double, #utb
                            c_double, #dpsi
                            c_double, #deps
                            c_double, #xp
                            c_double, #yp
                            ndpointer(shape=(3,3), dtype=float)] #rc2t
def C2tpe(tta, ttb, uta, utb, dpsi, deps, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1,
    the nutation and the polar motion. IAU 2000.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param dpsi, deps: nutation
    :type dpsi, deps: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: the celestial-to-terrestrial matrix as a nump.matrix of shape
        3x3.

    *sofa manual.pdf page 48*
    """
    rc2t = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2tpe(tta, ttb, uta, utb, float(dpsi), float(deps), float(xp),
                                                            float(yp), rc2t)
    return rc2t



# iauC2txy
sofa.iauC2txy.argtypes = [c_double, #tta
                            c_double, #ttb
                            c_double, #uta
                            c_double, #utb
                            c_double, #x
                            c_double, #y,
                            c_double, #xp,
                            c_double, #yp
                            ndpointer(shape=(3,3), dtype=float)] #rc2t
def C2txy(tta, ttb, uta, utb, x, y, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1,
    the CIP coordinates and the polar motion. IAU 2000.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param x, y: Celestial Intermediate Pole.
    :type x, y: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: celestial-to-terrestrial matrix as a numpy.matrix of shape
        3x3.

    *sofa manual.pdf page 50*
    """
    rc2t = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauC2txy(tta, ttb, uta, utb, float(x), float(y),
                                                float(xp), float(yp), rc2t)
    return rc2t


# iauCal2jd
sofa.iauCal2jd.argtypes = [c_int, #iy
                            c_int, #im
                            c_int, #id
                            POINTER(c_double), #djm0
                            POINTER(c_double)] #djm
sofa.iauCal2jd.restype = c_int
def Cal2jd(iy, im, id):
    """ Gregorian calendar to Julian date.

    :param iy: year.
    :type iy: int

    :param im: month.
    :type im: int

    :param id: day.
    :type id: int

    :returns: a tuple of three items:

        * MJD zero-point : always 2400000.5 (float)
        * Modified Julian date for 0 hours (float)
        * Function status.

    *sofa manual.pdf page 52*
    """
    djm0 = c_double()
    djm = c_double()
    status = sofa.iauCal2jd(iy, im, id, byref(djm0), byref(djm))
    return djm0.value, djm.value, status


# iauCp
sofa.iauCp.argtypes = [ndpointer(shape=(1,3), dtype=float), #p
                        ndpointer(shape=(1,3), dtype=float)] #c
def Cp(p):
    """ Copy a p-vector.

    :param p: p-vector to copy.
    :type p: array-like of shape (1,3)

    :returns: a copy of *p* as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 53*
    """

    c = asmatrix(zeros(shape=(1,3)), dtype=float)
    sofa.iauCp(asmatrix(p, dtype=float), c)
    return c


# iauCpv
sofa.iauCpv.argtypes = [ndpointer(shape=(2,3), dtype=float), #pv
                        ndpointer(shape=(2,3), dtype=float)] #c
def Cpv(pv):
    """ Copy a pv-vector.

    :param pv: pv-vector to copy.
    :type pv: array-like of shape (2,3)

    :returns: a copy of *pv* as a numpy.matrix of shape 2x3.

    *sofa manual.pdf page 54*
    """

    c = asmatrix(zeros(shape=(2,3)), dtype=float)
    sofa.iauCpv(asmatrix(pv, dtype=float), c)
    return c


# iauCr
sofa.iauCr.argtypes = [ndpointer(shape=(3,3), dtype=float), #r
                        ndpointer(shape=(3,3), dtype=float)] #c
def Cr(r):
    """ Copy a rotation matrix.

    :param r: rotation matrix to copy.
    :type r: array-like of shape (3,3)

    :returns: a copy of *r* as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 55*
    """

    c = asmatrix(zeros(shape=(3,3)), dtype=float)
    sofa.iauCr(asmatrix(r, dtype=float), c)
    return c


sofa.iauD2dtf.argtypes = [POINTER(c_char), #scale
                            c_int, #ndp
                            c_double, #d1
                            c_double, #d2
                            POINTER(c_int), #iy
                            POINTER(c_int), #im
                            POINTER(c_int), #id
                            c_int * 4] #ihmsf
sofa.iauD2dtf.restype = c_int
def D2dtf(scale, ndp, d1, d2):
    """ Format for output a 2-part Julian Date.

    :param scale: timescale ID.
    :type scale: str

    :param ndp: resolution.
    :type ndp: int

    :param d1, d2: time as a two-part Julian Date.
    :type d1, d2: float

    :returns: a tuple of 8 items:

        * year (int)
        * month (int)
        * day (int)
        * hours (int)
        * minutes (int)
        * seconds (int)
        * fraction of second (int)
        * function status.

    *sofa manual.pdf page 56*
    """
    iy = c_int()
    im = c_int()
    id = c_int()
    ihmsf = (c_int * 4)()
    s = sofa.iauD2dtf(scale, ndp, d1, d2, byref(iy), byref(im), byref(id), ihmsf)
    return (iy.value, im.value, id.value) + tuple([v for v in ihmsf]) + (s,)




# iauD2tf
sofa.iauD2tf.argtypes = [c_int, #ndp
                            c_double, #days
                            POINTER(c_char), #sign
                            c_int * 4] #ihmsf
def D2tf(ndp, days):
    """ Decompose days into hours, minutes, seconds, fraction.

    :param ndp: the requested resolution.
    :type ndp: int

    :param days: the value to decompose.
    :type days: float

    :returns: a tuple of two items:

        * the sign as a string ('+' or '-')
        * a tuple (hours, minutes, seconds, fraction).

    *sofa manual.pdf page 58*
    """
    sign = c_char()
    ihmsf = (c_int * 4)()
    sofa.iauD2tf(ndp, days, byref(sign), ihmsf)
    return sign.value, tuple([v for v in ihmsf])


# iauDat
sofa.iauDat.argtypes = [c_int, #iy
                        c_int, #im
                        c_int, #id
                        c_double, #fd
                        POINTER(c_double)] #deltat
sofa.iauDat.restype = c_int
def Dat(iy, im, id, fd):
    """ Calculate delta(AT) = TAI - UTC for a given UTC date.

    :param iy: UTC year.
    :type iy: int

    :param im: month.
    :type im: int

    :param id: day.
    :type id: int

    :param fd: fraction of day.
    :type fd: float

    :returns: deltat as a float.

    *sofa manual.pdf page 59*
    """
    deltat = c_double()
    status = sofa.iauDat(iy, im, id, fd, byref(deltat))
    return deltat.value, status


# iauDtdb
sofa.iauDtdb.argtypes = [c_double, #date1,
                            c_double, #date2
                            c_double, #ut
                            c_double, #elong
                            c_double, #u
                            c_double] #v
sofa.iauDtdb.restype = c_double
def Dtdb(date1, date2, ut, elong, u, v):
    """ Approximation of TDB - TT, the difference between barycentric dynamical
    time and terrestrial time, for an observer on Earth.

    :param date1, date2: TDB as a two-part date.
    :type date1, date2: float

    :param ut: universal time (UT1, fraction of one day).
    :type ut: float

    :param elong: longitude in radians (east positive)
    :type elong: float

    :param u: distance from Earth's spin axis in kilometers.
    :type u: float

    :param v: distance north of equatorial plane in kilometers
    :type v: float

    :returns: TDB - TT in seconds (float)

    *sofa manual.pdf page 61*
    """
    return sofa.iauDtdb(date1, date2, ut, float(elong), u, v)


# iauDtf2d
sofa.iauDtf2d.argtypes = [POINTER(c_char), #scale
                            c_int, #iy
                            c_int, #im
                            c_int, #id
                            c_int, #ihr
                            c_int, #imn
                            c_double, #sec
                            POINTER(c_double), #d1
                            POINTER(c_double)] #d2
sofa.iauDtf2d.restype = c_int
def Dtf2d(scale, iy, im, id, ihr, imn, sec):
    """ Encode date and time fields into a two-part Julian Date.

    :param scale: Timescale id.
    :type scale: str

    :param iy: year.
    :type iy: int

    :param im: month.
    :type im: int

    :param id: day.
    :type id: int

    :param ihr: hour.
    :type ihr: int

    :param imn: minute.
    :type imn: int

    :param sec: seconds.
    :type sec: float

    :returns: a tuple of three items:

        * the two-part Julian Date
        * function status.

    *sofa manual.pdf page 64*
    """

    d1 = c_double()
    d2 = c_double()
    s = sofa.iauDtf2d(scale, iy, im, id, ihr, imn, sec, byref(d1), byref(d2))
    return d1.value, d2.value, s



# iauEe00
sofa.iauEe00.argtypes = [c_double, #date1
                            c_double, #date2
                            c_double, # epsa
                            c_double] #dpsi
sofa.iauEe00.restype = c_double
def Ee00(date1, date2, epsa, dpsi):
    """ The equation of the equinoxes, compatible with IAU 2000 resolutions,
    given the nutation in longitude and the mean obliquity.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param epsa: mean obliquity.
    :type epsa: float

    :param dpsi: nutation in longitude.
    :type dpsi: float

    :returns: equation of the equinoxes (float).

    *sofa manual.pdf page 66*
    """
    return sofa.iauEe00(date1, date2, float(epsa), float(dpsi))

# iauEe00a
sofa.iauEe00a.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauEe00a.restype = c_double
def Ee00a(date1, date2):
    """ Equation of the equinoxes, compatible with IAU 2000 resolutions.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the equinoxes (float)

    *sofa manual.pdf page 67*
    """
    return sofa.iauEe00a(date1, date2)


# iauEe00b
sofa.iauEe00b.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauEe00b.restype = c_double
def Ee00b(date1, date2):
    """ Equation of the equinoxes, compatible with IAU 2000 resolutions, using
    truncated nutation model IAU 2000B.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the equinoxes (float)

    *sofa manual.pdf page 68*
    """
    return sofa.iauEe00b(date1, date2)


# iauEe06a
sofa.iauEe06a.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauEe06a.restype = c_double
def Ee06a(date1, date2):
    """ Equation of the equinoxes, compatible with IAU 2000 resolutions and
    IAU 2006/2000A precession-nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the equinoxes (float)

    *sofa manual.pdf page 69*
    """
    return sofa.iauEe06a(date1, date2)


# iauEect00
sofa.iauEect00.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauEect00.restype = c_double
def Eect00(date1, date2):
    """ Equation of the equinoxes complementary terms, consistent with IAU
    2000 resolutions.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: complementary terms (float).

    *sofa manual.pdf page 70*
    """
    return sofa.iauEect00(date1, date2)


# iauEform
sofa.iauEform.argtypes = [c_int, #n
                            POINTER(c_double), #a
                            POINTER(c_double)] #f
sofa.iauEform.restype = c_int
def Eform(n):
    """ Earth's reference ellipsoids.

    :param n: ellipsoid identifier, should be one of:

        #. WGS84
        #. GRS80
        #. WGS72
    :type n: int

    :returns: a tuple of three items:

        * equatorial radius in meters (float)
        * flattening (float)
        * function status.

    *sofa manual.pdf page 72*
    """
    a = c_double()
    f = c_double()
    status = sofa.iauEform(n, byref(a), byref(f))
    return a.value, f.value, status


# iauEo06a
sofa.iauEo06a.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauEo06a.restype = c_double
def Eo06a(date1, date2):
    """ Equation of the origins, IAU 2006 precession and IAU 2000A nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the origins in radians (float).

    *sofa manual.pdf page 73*
    """
    return sofa.iauEo06a(date1, date2)


# iauEors
sofa.iauEors.argtypes = [ndpointer(shape=(3,3), dtype=float), #rnpb
                            c_double] #s
sofa.iauEors.restype = c_double
def Eors(rnpb, s):
    """ Equation of the origins, given the classical NPB matrix and the
    quantity s.

    :param rnpb: classical nutation x precession x bias matrix.
    :type rnpb: array-like of shape (3,3)

    :param s: the CIO locator.
    :type s: float

    :returns: the equation of the origins in radians (float).

    *sofa manual.pdf page 74*
    """
    return sofa.iauEors(asmatrix(rnpb, dtype=float), float(s))


# iauEpb
sofa.iauEpb.argtypes = [c_double, #dj1
                        c_double] #dj2
sofa.iauEpb.restype = c_double
def Epb(dj1, dj2):
    """ Julian date to Besselian epoch.

    :param dj1, dj2: two-part Julian date.
    :type date1, date2: float

    :returns: Besselian epoch (float).

    *sofa manual.pdf page 75*
    """
    return sofa.iauEpb(dj1, dj2)


# iauEpb2jd
sofa.iauEpb2jd.argtypes = [c_double, #epb
                            POINTER(c_double), #djm0
                            POINTER(c_double)] #djm
def Epb2jd(epb):
    """ Besselian epoch to Julian date.

    :param epb: Besselian epoch.
    :type epb: float

    :returns: a tuple of two items:

        * MJD zero-point, always 2400000.5 (float)
        * modified Julian date (float).

    *sofa manual.pdf page 76*
    """
    djm0 = c_double()
    djm = c_double()
    sofa.iauEpb2jd(epb, byref(djm0), byref(djm))
    return djm0.value, djm.value


# iauEpj
sofa.iauEpj.argtypes = [c_double, #dj1
                        c_double] #dj2
sofa.iauEpj.restype = c_double
def Epj(dj1, dj2):
    """ Julian date to Julian epoch.

    :param dj1, dj2: two-part Julian date.
    :type dj1, dj2: float

    :returns: Julian epoch (float)

    *sofa manual.pdf page 77*
    """
    return sofa.iauEpj(dj1, dj2)


# iauEpj2jd
sofa.iauEpj2jd.argtypes = [c_double, #epj
                POINTER(c_double), #djm0
                POINTER(c_double)] #djm
def Epj2jd(epj):
    """ Julian epoch to Julian date.

    :param epj: Julian epoch.
    :type epj: float

    :returns: a tuple of two items:

        * MJD zero-point, always 2400000.5 (float)
        * modified Julian date (float).

    *sofa manual.pdf page 78*
    """
    djm0 = c_double()
    djm = c_double()
    sofa.iauEpj2jd(epj, byref(djm0), byref(djm))
    return djm0.value, djm.value


# iauEpv00
sofa.iauEpv00.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(2,3), dtype=float), #pvh
                            ndpointer(shape=(2,3), dtype=float)] # pvb
sofa.iauEpv00.restype = c_int
def Epv00(date1, date2):
    """ Earth position and velocity, heliocentric and barycentric, with
    respect to the Barycentric Celestial Reference System.

    :param date1, date2: TDB as a two-part Julian date.
    :type date1, date2: float

    :returns: a tuple of three items:

        * heliocentric Earth position velocity as a numpy.matrix of shape \
           2x3.
        * barycentric Earth position/velocity as a numpy.matrix of shape \
           2x3.
        * function status.

    *sofa manual.pdf page 79*
    """
    pvh = asmatrix(zeros(shape=(2,3), dtype=float))
    pvb = asmatrix(zeros(shape=(2,3), dtype=float))
    status = sofa.iauEpv00(date1, date2, pvh, pvb)
    return pvh, pvb, status


# iauEqeq94
sofa.iauEqeq94.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauEqeq94.restype = c_double
def Eqeq94(date1, date2):
    """ Equation of the equinoxes, IAU 1994 model.

    :param date1, date2: TDB as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the equinoxes (float).

    *sofa manual.pdf page 81*
    """
    return sofa.iauEqeq94(date1, date2)


# iauEra00
sofa.iauEra00.argtypes = [c_double, #dj1
                            c_double] #dj2
sofa.iauEra00.restype = c_double
def Era00(dj1, dj2):
    """ Earth rotation angle IAU 2000 model.

    :param dj1, dj2: UT1 as a two-part Julian date.
    :type dj1, dj2: float

    :returns: Earth rotation angle in radians, in the range 0-2pi (float).

    *sofa manual.pdf page 82*
    """
    return sofa.iauEra00(dj1, dj2)


# iauFad03
sofa.iauFad03.argtypes = [c_double] #t
sofa.iauFad03.restype = c_double
def Fad03(t):
    """ Mean elongation of the Moon from the Sun (fundamental argument, IERS
    conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean elongation of the Moon from the Sun in radians (float).

    *sofa manual.pdf page 83*
    """
    return sofa.iauFad03(t)


# iauFae03
sofa.iauFae03.argtypes = [c_double] #t
sofa.iauFae03.restype = c_double
def Fae03(t):
    """ Mean longitude of Earth (fundamental argument, IERS conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Earth in radians (float).

    *sofa manual.pdf page 84*
    """
    return sofa.iauFae03(t)


# iauFaf03
sofa.iauFaf03.argtypes = [c_double] #t
sofa.iauFaf03.restype = c_double
def Faf03(t):
    """ Mean longitude of the Moon minus mean longitude of the ascending node
    (fundamental argument, IERS conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: result in radians (float).

    *sofa manual.pdf page 85*
    """
    return sofa.iauFaf03(t)


# iauFaju03
sofa.iauFaju03.argtypes = [c_double] #t
sofa.iauFaju03.restype = c_double
def Faju03(t):
    """ Mean longitude of Jupiter (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Jupiter in radians (float).

    *sofa manual.pdf page 86*
    """
    return sofa.iauFaju03(t)


# iauFal03
sofa.iauFal03.argtypes = [c_double] #t
sofa.iauFal03.restype = c_double
def Fal03(t):
    """ Mean anomaly of the Moon (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean anomaly of the Moon in radians (float).

    *sofa manual.pdf page 87*
    """
    return sofa.iauFal03(t)


# iauFalp03
sofa.iauFalp03.argtypes = [c_double] #t
sofa.iauFalp03.restype = c_double
def Falp03(t):
    """ Mean anomaly of the Sun (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean anomaly of the Sun in radians (float).

    *sofa manual.pdf page 88*
    """
    return sofa.iauFalp03(t)


# iauFama03
sofa.iauFama03.argtypes = [c_double] #t
sofa.iauFama03.restype = c_double
def Fama03(t):
    """ Mean longitude of Mars (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Mars in radians (float).

    *sofa manual.pdf page 89*
    """
    return sofa.iauFama03(t)


# iauFame03
sofa.iauFame03.argtypes = [c_double] #t
sofa.iauFame03.restype= c_double
def Fame03(t):
    """ Mean longitude of Mercury (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Mercury in radians (float).

    *sofa manual.pdf page 90*
    """
    return sofa.iauFame03(t)


# iauFane03
sofa.iauFane03.argtypes = [c_double] #t
sofa.iauFane03.restype = c_double
def Fane03(t):
    """ Mean longitude of Neptune (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Neptune in radians (float).

    *sofa manual.pdf page 91*
    """
    return sofa.iauFane03(t)


# iauFaom03
sofa.iauFaom03.argtypes = [c_double] #t
sofa.iauFaom03.restype = c_double
def Faom03(t):
    """ Mean longitude of the Moon's ascending node (fundamental argument,
    IERS conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of of the Moon's ascending node in radians
        (float).

    *sofa manual.pdf page 92*
    """
    return sofa.iauFaom03(t)


# iauFapa03
sofa.iauFapa03.argtypes = [c_double] #t
sofa.iauFapa03.restype = c_double
def Fapa03(t):
    """ General accumulated precession in longitude (fundamental argument,
    IERS conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: general accumulated precession in longitude in radians
        (float).

    *sofa manual.pdf page 93*
    """
    return sofa.iauFapa03(t)


# iauFasa03
sofa.iauFasa03.argtypes = [c_double] #t
sofa.iauFasa03.restype = c_double
def Fasa03(t):
    """ Mean longitude of Saturn (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Saturn in radians (float).

    *sofa manual.pdf page 94*
    """
    return sofa.iauFasa03(t)


# iauFaur03
sofa.iauFaur03.argtypes = [c_double] #t
sofa.iauFaur03.restype = c_double
def Faur03(t):
    """ Mean longitude of Uranus (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Uranus in radians (float).

    *sofa manual.pdf page 95*
    """
    return sofa.iauFaur03(t)


# iauFave03
sofa.iauFave03.argtypes = [c_double] #t
sofa.iauFave03.restype = c_double
def Fave03(t):
    """ Mean longitude of Venus (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Venus in radians (float).

    *sofa manual.pdf page 96*
    """
    return sofa.iauFave03(t)


# iauFk52h
sofa.iauFk52h.argtypes = [c_double, #r5
                            c_double, #d5
                            c_double, #dr5
                            c_double, #dd5
                            c_double, #px5
                            c_double, #rv5
                            POINTER(c_double), #rh
                            POINTER(c_double), #dh
                            POINTER(c_double), #drh
                            POINTER(c_double), #ddh
                            POINTER(c_double), #pxh
                            POINTER(c_double)] #rvh
def Fk52h(r5, d5, dr5, dd5, px5, rv5):
    """ Transform FK5 (J2000.0) star data into the Hipparcos system.

    :param r5: right ascension in radians.
    :type r5: float

    :param d5: declination in radians.
    :type d5: float

    :param dr5: proper motion in RA (dRA/dt, rad/Jyear).
    :type dr5: float

    :param dd5: proper motion in Dec (dDec/dt, rad/Jyear).
    :type dd5: float

    :param px5: parallax (arcseconds)
    :type px5: float

    :param rv5: radial velocity (km/s, positive = receding)
    :type rv5: float

    :returns: a tuple of six items corresponding to Hipparcos epoch J2000.0:

        * right ascension
        * declination
        * proper motion in RA (dRa/dt, rad/Jyear)
        * proper motion in Dec (dDec/dt, rad/Jyear)
        * parallax in arcseconds
        * radial velocity (km/s, positive = receding).

    *sofa manual.pdf page 97*
    """
    rh = c_double()
    dh = c_double()
    drh = c_double()
    ddh = c_double()
    pxh = c_double()
    rvh = c_double()
    sofa.iauFk52h(float(r5), float(d5), float(dr5), float(dd5), float(px5),
                    float(rv5),
                    byref(rh), byref(dh), byref(drh), byref(ddh),
                    byref(pxh), byref(rvh))
    return rh.value, dh.value, drh.value, ddh.value, pxh.value, rvh.value


# iauFk5hip
sofa.iauFk5hip.argtypes = [ndpointer(shape=(3,3), dtype=float), #r5h
                            ndpointer(shape=(1,3), dtype=float)] #s5h
def Fk5hip():
    """ FK5 to Hipparcos rotation and spin.

    :returns: a tuple of two items:

        * FK5 rotation wrt Hipparcos as a numpy.matrix of shape 3x3
        * FK5 spin wrt Hipparcos as a numpy.matrix of shape 1x3

    *sofa manual.pdf page 98*
    """
    r5h = asmatrix(zeros(shape=(3,3), dtype=float))
    s5h = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauFk5hip(r5h, s5h)
    return r5h, s5h


# iauFk5hz
sofa.iauFk5hz.argtypes = [c_double, #r5
                            c_double, #d5
                            c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #rh
                            POINTER(c_double)] #dh
def Fk5hz(r5, d5, date1, date2):
    """ Transform an FK5 (J2000.0) star position into the system of the
    Hipparcos catalogue, assuming zero Hipparcos proper motion.

    :param r5: right ascension in radians, equinox J2000.0, at date.
    :type r5: float

    :param d5: declination in radians, equinox J2000.0, at date.
    :type d5: float

    :param date1, date2: TDB date as a two-part Julian date.
    :type date1, date2: float

    :returns: a tuple of two items:

        * Hipparcos right ascension in radians (float)
        * Hipparcos declination in radians (float).

    *sofa manual.pdf page 99*
    """
    rh = c_double()
    dh = c_double()
    sofa.iauFk5hz(float(r5), float(d5), date1, date2, byref(rh), byref(dh))
    return rh.value, dh.value


# iauFw2m
sofa.iauFw2m.argtypes = [c_double, #gamb
                            c_double, #phib
                            c_double, #psi
                            c_double, #eps
                            ndpointer(shape=(3,3), dtype=float)] #r
def Fw2m(gamb, phib, psi, eps):
    """ Form rotation matrix given the Fukushima-Williams angles.

    :param gamb: F-W angle gamma_bar in radians.
    :type gamb: float

    :param phib: F-W angle phi_bar in radians.
    :type phib: float

    :param psi: F-W angle psi in radians.
    :type psi: float

    :param eps: F-W angle epsilon in radians.
    :type epsilon: float

    :returns: rotation matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 101*
    """
    r = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauFw2m(float(gamb), float(phib), float(psi), float(eps), r)
    return r


# iauFw2xy
sofa.iauFw2xy.argtypes = [c_double, #gamb
                            c_double, #phib
                            c_double, #psi
                            c_double, #eps
                            POINTER(c_double), #x
                            POINTER(c_double)] #y
def Fw2xy(gamb, phib, psi, eps):
    """ CIP X and Y given Fukushima-Williams bias-precession-nutation angles.

    :param gamb: F-W angle gamma_bar in radians.
    :type gamb: float

    :param phib: F-W angle phi_bar in radians.
    :type phib: float

    :param psi: F-W angle psi in radians.
    :type psi: float

    :param eps: F-W angle epsilon in radians.
    :type epsilon: float

    :returns: a tuple containing CIP X and X in radians (float).

    *sofa manual.pdf page 103*
    """
    x = c_double()
    y = c_double()
    sofa.iauFw2xy(float(gamb), float(phib), float(psi), float(eps),
                                                        byref(x), byref(y))
    return x.value, y.value


# iauGc2gd
sofa.iauGc2gd.argtypes = [c_int, #n
                            ndpointer(shape=(1,3), dtype=float), #xyz
                            POINTER(c_double), #elong
                            POINTER(c_double), #phi
                            POINTER(c_double)] #height
sofa.iauGc2gd.restype = c_int
def Gc2gd(n, xyz):
    """ Transform geocentric coordinates to geodetic using the specified
    reference ellipsoid.

    :param n: ellipsoid identifier, should be one of:

        #. WGS84
        #. GRS80
    :type n: int

    :param xyz: geocentric vector.
    :type xyz: array-like of shape (1,3)

    :returns: a tuple of four items:

        * longitude in radians (float)
        * geodetic latitude in radians (float)
        * geodetic height above ellipsoid (float)
        * function status.

    *sofa manual.pdf page 104*
    """
    elong = c_double()
    phi = c_double()
    height = c_double()
    status = sofa.iauGc2gd(n, asmatrix(xyz, dtype=float), byref(elong),
                            byref(phi), byref(height))
    return elong.value, phi.value, height.value, status


# iauGc2gde
sofa.iauGc2gde.argtypes = [c_double, #a
                            c_double, #f
                            ndpointer(shape=(1,3), dtype=float), #xyz
                            POINTER(c_double), #elong
                            POINTER(c_double), #phi
                            POINTER(c_double)] #height
sofa.iauGc2gde.restype = c_int
def Gc2gde(a, f, xyz):
    """ Transform geocentric coordinates to geodetic for a reference
    ellipsoid of specified form.

    :param a: equatorial radius.
    :type a: float

    :param f: flattening.
    :type f: float

    :param xyz: geocentric vector.
    :type xyz: array-like of shape (1,3)

    :returns: a tuple of four items:

        * longitude in radians
        * geodetic latitude in radians
        * geodetic height above ellipsoid
        * function status.

    *sofa manual.pdf page 105*
    """
    elong = c_double()
    phi = c_double()
    height = c_double()
    status = sofa.iauGc2gde(a, f, asmatrix(xyz, dtype=float), byref(elong),
                                byref(phi), byref(height))
    return elong.value, phi.value, height.value, status


# iauGd2gc
sofa.iauGd2gc.argtypes = [c_int, #n
                            c_double, #elong
                            c_double, #phi,
                            c_double, #height
                            ndpointer(shape=(1,3), dtype=float)] #xyz
sofa.iauGd2gc.restype = c_int
def Gd2gc(n, elong, phi, height):
    """ Transform geodetic coordinates to geocentric using specified reference
    ellipsoid.

    :param n: ellipsoid identifier, should be one of:

        #. WGS84
        #. GRS80
    :type n: int

    :param elong: longitude in radians.
    :type elong: float

    :param phi: geodetic latitude in radians.
    :type phi: float

    :param height: geodetic height above ellipsoid in meters.
    :type height: float

    :returns: geocentric vector as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 106*
    """
    xyz = asmatrix(zeros(shape=(1,3), dtype=float))
    status = sofa.iauGd2gc(n, float(elong), float(phi), height, xyz)
    return xyz, status


# iauGd2gce
sofa.iauGd2gce.argtypes = [c_double, #a
                            c_double, #f
                            c_double, #elong
                            c_double, #phi
                            c_double, #height
                            ndpointer(shape=(1,3), dtype=float)] #xyz
sofa.iauGd2gce.restype = c_int
def Gd2gce(a, f, elong, phi, height):
    """ Transform geodetic coordinates to geocentric for a reference
    ellipsoid of specified form.

    :param a: equatorial radius.
    :type a: float

    :param f: flattening.
    :type f: float

    :param elong: longitude in radians.
    :type elong: float

    :param phi: geodetic latitude in radians.
    :type phi: float

    :param height: geodetic height above ellipsoid in meters.
    :type height: float

    :returns: geocentric vector as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 107*
    """
    xyz = asmatrix(zeros(shape=(1,3), dtype=float))
    status = sofa.iauGd2gce(a, f, float(elong), float(phi), height, xyz)
    return xyz, status


# iauGmst00
sofa.iauGmst00.argtypes = [c_double, #uta
                            c_double, #utb
                            c_double, #tta
                            c_double] #ttb
sofa.iauGmst00.restype = c_double
def Gmst00(uta, utb, tta, ttb):
    """ Greenwich mean sidereal time, consistent with IAU 2000 resolutions.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :returns: Greenwich mean sidereal time in radians (float).

    *sofa manual.pdf page 108*
    """
    return sofa.iauGmst00(uta, utb, tta, ttb)


# iauGmst06
sofa.iauGmst06.argtypes = [c_double, #uta
                            c_double, #utb
                            c_double, #tta
                            c_double] #ttb
sofa.iauGmst06.restype = c_double
def Gmst06(uta, utb, tta, ttb):
    """ Greenwich mean sidereal time, consistent with IAU 2006 precession.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :returns: Greenwich mean sidereal time in radians (float).

    *sofa manual.pdf page 110*
    """
    return sofa.iauGmst06(uta, utb, tta, ttb)


# iauGmst82
sofa.iauGmst82.argtypes = [c_double, #dj1
                            c_double] #dj2
sofa.iauGmst82.restype = c_double
def Gmst82(dj1, dj2):
    """ Greenwich mean sidereal time, IAU 1982 model.

    :param dj1, dj2: UT1 as a two-part Julian date.
    :type uta, utb: float

    :returns: Greenwich mean sidereal time in radians (float).

    *sofa manual.pdf page 111*
    """
    return sofa.iauGmst82(dj1, dj2)


# iauGst00a
sofa.iauGst00a.argtypes = [c_double, #uta
                            c_double, #utb
                            c_double, #tta
                            c_double] #ttb
sofa.iauGst00a.restype = c_double
def Gst00a(uta, utb, tta, ttb):
    """ Greenwich apparent sidereal time, consistent with IAU 2000 resolutions.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :returns: Greenwich apparent sidereal time in radians (float).

    *sofa manual.pdf page 112*
    """
    return sofa.iauGst00a(uta, utb, tta, ttb)


# iauGst00b
sofa.iauGst00b.argtypes = [c_double, #uta
                            c_double] #utb
sofa.iauGst00b.restype = c_double
def Gst00b(uta, utb):
    """ Greenwich apparent sidereal time, consistent with IAU 2000 resolutions,
    using truncated nutation model IAU 2000B.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :returns: Greenwich apparent sidereal time in radians (float).

    *sofa manual.pdf page 114*
    """
    return sofa.iauGst00b(uta, utb)


# iauGst06
sofa.iauGst06.argtypes = [c_double, #uta
                            c_double, #utb
                            c_double, #tta
                            c_double, #ttb
                            ndpointer(shape=(3,3), dtype=float)] #rnpb
sofa.iauGst06.restype = c_double
def Gst06(uta, utb, tta, ttb, rnpb):
    """ Greenwich apparent sidereal time, IAU 2006, given the *npb* matrix.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param rnpb: nutation x precession x bias matrix.
    :type rnpb: array-like of shape (3,3)

    :returns: Greenwich apparent sidereal time in radians (float).

    *sofa manual.pdf page 116*
    """
    return sofa.iauGst06(uta, utb, tta, ttb, asmatrix(rnpb, dtype=float))


# iauGst06a
sofa.iauGst06a.argtypes = [c_double, #uta
                            c_double, #utb
                            c_double, #tta
                            c_double] #ttb
sofa.iauGst06a.restype = c_double
def Gst06a(uta, utb, tta, ttb):
    """ Greenwich apparent sidereal time, consistent with IAU 2000 and 2006
    resolutions.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :returns: Greenwich apparent sidereal time in radians (float).

    *sofa manual.pdf page 117*
    """
    return sofa.iauGst06a(uta, utb, tta, ttb)


# iauGst94
sofa.iauGst94.argtypes = [c_double, #uta
                            c_double] #utb
sofa.iauGst94.restype = c_double
def Gst94(uta, utb):
    """ Greenwich apparent sidereal time, consistent with IAU 1982/94
    resolutions.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :returns: Greenwich apparent sidereal time in radians (float).

    *sofa manual.pdf page 118*
    """
    return sofa.iauGst94(uta, utb)


# iauH2fk5
sofa.iauH2fk5.argtypes = [c_double, #rh
                            c_double, #dh
                            c_double, #drh
                            c_double, #ddh
                            c_double, #pxh
                            c_double, #rvh
                            POINTER(c_double), #r5
                            POINTER(c_double), #d5
                            POINTER(c_double), #dr5
                            POINTER(c_double), #dd5
                            POINTER(c_double), #px5
                            POINTER(c_double)] #rv5
def H2fk5(rh, dh, drh, ddh, pxh, rvh):
    """ Transform Hipparcos star data into FK5 (J2000.0) system.

    :param rh: right ascension in radians.
    :type rh: float

    :param dh: declination in radians.
    :type dh: float

    :param drh: proper motion in RA (dRA/dt, rad/Jyear).
    :type drh: float

    :param ddh: proper motion in Dec (dDec/dt, rad/Jyear).
    :type ddh: float

    :param pxh: parallax in arcseconds.
    :type pxh: float

    :param rvh: radial velocity (km/s, positive = receding).
    :type rvh: float

    :returns: a tuple of six items:

        * right ascension in radians
        * declination in radians
        * proper motion in RA (dRA/dt, rad/Jyear)
        * proper motion in Dec (dDec/dt, rad/Jyear)
        * parallax in arcseconds
        * radial velocity (km/s, positive = receding).

    *sofa manual.pdf page 119*
    """
    r5 = c_double()
    d5 = c_double()
    dr5 = c_double()
    dd5 = c_double()
    px5 = c_double()
    rv5 = c_double()
    sofa.iauH2fk5(float(rh), float(dh), float(drh), float(ddh),
                    float(pxh), float(rvh), byref(r5), byref(d5),
                    byref(dr5), byref(dd5), byref(px5),byref(rv5))
    return r5.value, d5.value, dr5.value, dd5.value, px5.value, rv5.value


# iauHfk5z
sofa.iauHfk5z.argtypes = [c_double, #rh
                            c_double, #dh
                            c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #r5
                            POINTER(c_double), #d5
                            POINTER(c_double), #dr5
                            POINTER(c_double)] #dd5
def Hfk5z(rh, dh, date1, date2):
    """ Transform Hipparcos star position into FK5 (J2000.0), assuming
    zero Hipparcos proper motion.

    :param rh: right ascension in radians.
    :type rh: float

    :param dh: declination in radians.
    :type dh: float

    :param date1, date2: TDB as a two-part Julian date.
    :type date1, date2: float

    :returns: a tuple of four items:

        * right ascension in radians
        * declination in radians
        * proper motion in RA (rad/year)
        * proper motion in Dec (rad/year)

    *sofa manual.pdf page 120*
    """
    r5 = c_double()
    d5 = c_double()
    dr5 = c_double()
    dd5 = c_double()
    sofa.iauHfk5z(float(rh), float(dh), date1, date2, byref(r5), byref(d5),
                                                    byref(dr5), byref(dd5))
    return r5.value, d5.value, dr5.value, dd5.value


# iauIr
sofa.iauIr.argtypes = [ndpointer(shape=(3,3), dtype=float)] #r
def Ir():
    """ Create a new rotation matrix initialized to the identity matrix.

    :returns: an identity matrix as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 122*
    """

    r = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauIr(r)
    return r


# iauJd2cal
sofa.iauJd2cal.argtypes = [c_double, #dj&
                            c_double, #dj2
                            POINTER(c_int), #iy
                            POINTER(c_int), #im
                            POINTER(c_int), #id
                            POINTER(c_double)] #fd
sofa.iauJd2cal.restype = c_int
def Jd2cal(dj1, dj2):
    """ Julian date to Gregorian year, month, day and fraction of day.

    :param dj1, dj2: two-part Julian date.
    :type dj1, dj2: float

    :returns: a tuple of five values:

        * year (int)
        * month (int)
        * day (int)
        * fraction of day (float)
        * function status

    *sofa manual.pdf page 123*
    """
    iy = c_int()
    im = c_int()
    id = c_int()
    fd = c_double()
    status = sofa.iauJd2cal(dj1, dj2, byref(iy), byref(im), byref(id),
                            byref(fd))
    return iy.value, im.value, id.value, fd.value, status


# iauJdcalf
sofa.iauJdcalf.argtypes = [c_int, #ndp
                            c_double, #dj1
                            c_double, #dj2
                            c_int * 4] #iymdf
sofa.iauJdcalf.restype = c_int
def Jdcalf(ndp, dj1, dj2):
    """ Julian date to Gregorian calendar, expressed in a form convenient
    for formatting messages: rounded to a specified precision.

    :param ndp: number of decimal places of days fraction.
    :type ndp: int

    :param dj1, dj2: two-part Julian date.
    :type dj1, dj2: float

    :returns: a tuple of two values:

        * a tuple wich itself contains (year, month, day, fraction)
        * function status.

    *sofa manual.pdf page 124*
    """
    iymdf = (c_int * 4)()
    status = sofa.iauJdcalf(ndp, dj1, dj2, iymdf)
    return tuple([v for v in iymdf]), status


# iauNum00a
sofa.iauNum00a.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rmatn
def Num00a(date1, date2):
    """ Form the matrix of nutation for a given date, IAU 2000A model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: nutation matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 125*
    """
    rmatn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauNum00a(date1, date2, rmatn)
    return rmatn


# iauNum00b
sofa.iauNum00b.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rmatn
def Num00b(date1, date2):
    """ Form the matrix of nutation for a given date, IAU 2000B model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: nutation matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 126*
    """
    rmatn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauNum00b(date1, date2, rmatn)
    return rmatn


# iauNum06a
sofa.iauNum06a.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rmatn
def Num06a(date1, date2):
    """ Form the matrix of nutation for a given date, IAU 2006/2000A model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: nutation matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 127*
    """
    rmatn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauNum06a(date1, date2, rmatn)
    return rmatn


# iauNumat
sofa.iauNumat.argtypes = [c_double, #epsa
                            c_double, #dpsi
                            c_double, #deps
                            ndpointer(shape=(3,3), dtype=float)] #rmatn
def Numat(epsa, dpsi, deps):
    """ Form the matrix of nutation.

    :param epsa: mean obliquity of date.
    :type epsa: float

    :param dpsi, deps: nutation.
    :type dpsi, deps: float

    :returns: nutation matrix as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 128*
    """
    rmatn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauNumat(float(epsa), float(dpsi), float(deps), rmatn)
    return rmatn


# iauNut00a
sofa.iauNut00a.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #dpsi
                            POINTER(c_double)] #deps
def Nut00a(date1, date2):
    """ Nutation, IAU 2000A model (MHB2000 luni-solar and planetary nutation
    with free core nutation omitted).

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * nutation in longitude in radians (float)
        * nutation in obliquity in radians (float).

    *sofa manual.pdf page 129*
    """
    dpsi = c_double()
    deps = c_double()
    sofa.iauNut00a(date1, date2, byref(dpsi), byref(deps))
    return dpsi.value, deps.value


# iauNut00b
sofa.iauNut00b.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #dpsi
                            POINTER(c_double)] #deps
def Nut00b(date1, date2):
    """ Nutation, IAU 2000B model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * nutation in longitude in radians (float)
        * nutation in obliquity in radians (float).

    *sofa manual.pdf page 132*
    """
    dpsi = c_double()
    deps = c_double()
    sofa.iauNut00b(date1, date2, byref(dpsi), byref(deps))
    return dpsi.value, deps.value


# iauNut06a
sofa.iauNut06a.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #dpsi
                            POINTER(c_double)] #deps
def Nut06a(date1, date2):
    """ IAU 2000A nutation with adjustments to match the IAU 2006 precession.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * nutation in longitude in radians (float)
        * nutation in obliquity in radians (float).

    *sofa manual.pdf page 134*
    """
    dpsi = c_double()
    deps = c_double()
    sofa.iauNut06a(date1, date2, byref(dpsi), byref(deps))
    return dpsi.value, deps.value


# iauNut80
sofa.iauNut80.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #dpsi
                            POINTER(c_double)] #deps
def Nut80(date1, date2):
    """ Nutation, IAU 1980 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * nutation in longitude in radians (float)
        * nutation in obliquity in radians (float).

    *sofa manual.pdf page 136*
    """
    dpsi = c_double()
    deps = c_double()
    sofa.iauNut80(date1, date2, byref(dpsi), byref(deps))
    return dpsi.value, deps.value


# iauNutm80
sofa.iauNutm80.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rmatn
def Nutm80(date1, date2):
    """ Form the nutation matrix for a given date, IAU 1980 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the nutation matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 137*
    """
    rmatn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauNutm80(date1, date2, rmatn)
    return rmatn


# iauObl06
sofa.iauObl06.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauObl06.restype = c_double
def Obl06(date1, date2):
    """ Mean obliquity of the ecliptic, IAU 2006 precession model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: obliquity of the ecliptic in radians (float).

    *sofa manual.pdf page 138*
    """
    return sofa.iauObl06(date1, date2)


# iauObl80
sofa.iauObl80.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauObl80.restype = c_double
def Obl80(date1, date2):
    """ Mean obliquity of the ecliptic, IAU 1980 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: obliquity of the ecliptic in radians (float).

    *sofa manual.pdf page 139*
    """
    return sofa.iauObl80(date1, date2)


# iauP06e
sofa.iauP06e.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #eps0
                            POINTER(c_double), #psia
                            POINTER(c_double), #oma
                            POINTER(c_double), #bpa
                            POINTER(c_double), #bqa
                            POINTER(c_double), #pia
                            POINTER(c_double), #bpia
                            POINTER(c_double), #epsa
                            POINTER(c_double), #chia
                            POINTER(c_double), #za
                            POINTER(c_double), #zetaa
                            POINTER(c_double), #thetaa
                            POINTER(c_double), #pa
                            POINTER(c_double), #gam
                            POINTER(c_double), #phi
                            POINTER(c_double)] #psi
def P06e(date1, date2):
    """ Precession angles, IAU 2006, equinox based.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 16-tuple:

        * epsilon_0
        * psi_A
        * omega_A
        * P_A
        * Q_A
        * pi_A
        * Pi_A
        * obliquity epsilon_A
        * chi_A
        * z_A
        * zeta_A
        * theta_A
        * p_A
        * F-W angle gamma_J2000
        * F-W angle phi_J2000
        * F-W angle psi_J2000

    *sofa manual.pdf page 140*
    """
    eps0 = c_double()
    psia = c_double()
    oma = c_double()
    bpa = c_double()
    bqa = c_double()
    pia = c_double()
    bpia = c_double()
    epsa = c_double()
    chia = c_double()
    za = c_double()
    zetaa = c_double()
    thetaa = c_double()
    pa = c_double()
    gam = c_double()
    phi = c_double()
    psi = c_double()
    sofa.iauP06e(date1, date2, byref(eps0), byref(psia), byref(oma),
                    byref(bpa), byref(bqa), byref(pia), byref(bpia),
                    byref(epsa), byref(chia), byref(za), byref(zetaa),
                    byref(thetaa), byref(pa), byref(gam), byref(phi),
                    byref(psi))
    return eps0.value, psia.value, oma.value, bpa.value, bqa.value, pia.value,\
            bpia.value, epsa.value, chia.value, za.value, zetaa.value, \
            thetaa.value, pa.value, gam.value, phi.value, psi.value


# iauP2pv
sofa.iauP2pv.argtypes = [ndpointer(shape=(1,3), dtype=float), #p
                            ndpointer(shape=(2,3), dtype=float)] #pv
def P2pv(p):
    """ Extend a p-vector to a pv-vector by appending a zero velocity.

    :param p: p-vector to extend.
    :type p: array-like of shape (1,3)

    :returns: pv-vector as a numpy.matrix of shape 2x3.

    *sofa manual.pdf page 142*
    """
    pv = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauP2pv(asmatrix(p, dtype=float), pv)
    return pv


# iauP2s
sofa.iauP2s.argtypes = [ndpointer(shape=(1,3), dtype=float), #p
                        POINTER(c_double), #theta
                        POINTER(c_double), #phi
                        POINTER(c_double)] #r
def P2s(p):
    """ P-vector to spherical polar coordinates.

    :param p: the p-vector
    :type p: array-like of shape (1,3)

    :returns: a 3-tuple:

        * longitude angle in radians (float)
        * latitude angle in radians (float)
        * radial distance (float).

    *sofa manual.pdf page 143*
    """
    theta = c_double()
    phi = c_double()
    r = c_double()
    sofa.iauP2s(asmatrix(p, dtype=float), theta, phi, r)
    return theta.value, phi.value, r.value


# iauPap
sofa.iauPap.argtypes = [ndpointer(shape=(1,3), dtype=float), #a
                        ndpointer(shape=(1,3), dtype=float)] #b
sofa.iauPap.restype = c_double
def Pap(a, b):
    """ Position-angle from two p-vectors.

    :param a: direction of the reference point.
    :type a: array-like of shape (1,3)

    :param b: direction of point whose position angle is required.
    :type b: array-like of shape (1,3)

    :returns: position angle of *b* with respect to *a* in radians (float).

    *sofa manual.pdf page 144*
    """
    return sofa.iauPap(asmatrix(a, dtype=float), asmatrix(b, dtype=float))


# iauPas
sofa.iauPas.argtypes = [c_double, #al
                        c_double, #ap
                        c_double, #bl
                        c_double] #bp
sofa.iauPas.restype = c_double
def Pas(al, ap, bl,bp):
    """ Postion-angle from spherical coordinates.

    :param al: longitude of point A in radians.
    :type al: float

    :param ap: latitude of point A in radians.
    :type ap: float

    :param bl: longitude of point B in radians.
    :type bl: float

    :param bp: latitude of point B in radians.
    :type bp: float

    :returns: position angle of B with respect to A in radians (float).

    *sofa manual.pdf page 145*
    """
    return sofa.iauPas(float(al), float(ap), float(bl), float(bp))


# iauPb06
sofa.iauPb06.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #bzeta
                            POINTER(c_double), #bz
                            POINTER(c_double)] #btheta
def Pb06(date1, date2):
    """ Form the three Euler angles which implement general precession from
    epoch J2000.0, using IAU 2006 model. Frame bias is included.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 3-tuple:

        * 1st rotation: radians cw around z (float)
        * 3rd rotation: radians cw around z (float)
        * 2nd rotation: radians ccw around y.

    *sofa manual.pdf page 146*
    """
    bzeta = c_double()
    bz = c_double()
    btheta = c_double()
    sofa.iauPb06(date1, date2, bzeta, bz, btheta)
    return bzeta.value, bz.value, btheta.value


# iauPdp
sofa.iauPdp.argtypes = [ndpointer(shape=(1,3), dtype=float), #a
                        ndpointer(shape=(1,3), dtype=float)] #b
sofa.iauPdp.restype = c_double
def Pdp(a, b):
    """ P-vector inner product.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a dot b as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 147*
    """
    return sofa.iauPdp(asmatrix(a, dtype=float), asmatrix(b, dtype=float))


# iauPfw06
sofa.iauPfw06.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #gamb
                            POINTER(c_double), #phib
                            POINTER(c_double), #psib
                            POINTER(c_double)] #epsa
def Pfw06(date1, date2):
    """ Precession angles, IAU 2006 (Fukushima-Williams 4-angle formulation).

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 4-tuple:

        * F-W angle gamma_bar in radians (float)
        * F-W angle phi_bar in radians (float)
        * F-W angle psi_bar in radians (float)
        * F-W angle epsilon_A in radians (float).

    *sofa manual.pdf page 148*
    """
    gamb = c_double()
    phib = c_double()
    psib = c_double()
    epsa = c_double()
    sofa.iauPfw06(date1, date2, gamb, phib, psib, epsa)
    return gamb.value, phib.value, psib.value, epsa.value


# iauPlan94
sofa.iauPlan94.argtypes = [c_double, #date1
                            c_double, #date2
                            c_int, #np
                            ndpointer(shape=(2,3), dtype=float)] #pv
sofa.iauPlan94.restype = c_int
def Plan94(date1, date2, np):
    """ Approximate heliocentric position and velocity of a nominated major
    planet : Mercury, Venus, EMB, Mars, Jupiter, Saturn, Uranus or Neptune.

    :param date1, date2: TDB as a two-part Julian date.
    :type date1, date2: float

    :param np: planet identifier (1=Mercury, 2=Venus, 3=EMB, 4=Mars, 5=Jupiter,
                                6=Saturn, 7=Uranus, 8=Neptune).
    :type np: int

    :returns: a 2-tuple:

        * planet's position and velocity (heliocentric, J2000.0, AU, AU/d) as \
            a numpy.matrix of shape 2x3
        * function status

    *sofa manual.pdf page 150*
    """
    pv = asmatrix(zeros(shape=(2,3), dtype=float))
    status = sofa.iauPlan94(date1, date2, np, pv)
    return pv, status


# iauPm
sofa.iauPm.argtypes = [ndpointer(shape=(1,3), dtype=float)] #p
sofa.iauPm.restype = c_double
def Pm(p):
    """ Modulus of p-vector.

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: modulus (float).

    *sofa manual.pdf page 153*
    """
    return sofa.iauPm(asmatrix(p, dtype=float))


# iauPmat00
sofa.iauPmat00.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rbp
def Pmat00(date1, date2):
    """ Precession matrix (including frame bias) from GCRS to a specified 
    date, IAU 2000 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 154*
    """
    rbp = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPmat00(date1, date2, rbp)
    return rbp


# iauPmat06
sofa.iauPmat06.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rbp
def Pmat06(date1, date2):
    """ Precession matrix (including frame bias) from GCRS to a specified 
    date, IAU 2006 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 155*
    """
    rbp = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPmat06(date1, date2, rbp)
    return rbp


# iauPmat76
sofa.iauPmat76.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rmatp
def Pmat76(date1, date2):
    """ Precession matrix from J2000.0 to a specified date, IAU 1976 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 156*
    """
    rmatp = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPmat76(date1, date2, rmatp)
    return rmatp


# iauPmp
sofa.iauPmp.argtypes = [ndpointer(shape=(1,3), dtype=float), #a
                        ndpointer(shape=(1,3), dtype=float), #b
                        ndpointer(shape=(1,3), dtype=float)] #amb
def Pmp(a, b):
    """ P-vector subtraction.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a - b as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 158*
    """
    amb = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauPmp(asmatrix(a, dtype=float), asmatrix(b, dtype=float), amb)
    return amb


# iauPn
sofa.iauPn.argtypes = [ndpointer(shape=(1,3), dtype=float), #p
                        POINTER(c_double), #r
                        ndpointer(shape=(1,3), dtype=float)] #u
def Pn(p):
    """ Convert a p-vector into modulus and unit vector.

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: 2-tuple:

            * the modulus (float)
            * unit vector (numpy.matrix of shape 1x3)

    *sofa manual.pdf page 159*
    """
    r = c_double()
    u = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauPn(asmatrix(p, dtype=float), byref(r), u)
    return r.value, u


# iauPn00
sofa.iauPn00.argtypes = [c_double, #date1
                            c_double, #date2
                            c_double, #dpsi
                            c_double, #deps
                            POINTER(c_double), #epsa
                            ndpointer(shape=(3,3), dtype=float), #rb
                            ndpointer(shape=(3,3), dtype=float), #rp
                            ndpointer(shape=(3,3), dtype=float), #rbp
                            ndpointer(shape=(3,3), dtype=float), #rn
                            ndpointer(shape=(3,3), dtype=float)] #rbpn
def Pn00(date1, date2, dpsi, deps):
    """ Precession-nutation, IAU 2000 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param dpsi, deps: nutation.
    :type dpsi, deps: float

    :returns: a 6-tuple:

            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    *sofa manual.pdf page 160*
    """
    epsa = c_double()
    rb = asmatrix(zeros(shape=(3,3), dtype=float))
    rp = asmatrix(zeros(shape=(3,3), dtype=float))
    rbp = asmatrix(zeros(shape=(3,3), dtype=float))
    rn = asmatrix(zeros(shape=(3,3), dtype=float))
    rbpn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPn00(date1, date2, float(dpsi), float(deps), byref(epsa),
                                                    rb, rp, rbp, rn, rbpn)
    return epsa.value, rb, rp, rbp, rn, rbpn


# iauPn00a
sofa.iauPn00a.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #dpsi
                            POINTER(c_double), #deps
                            POINTER(c_double), #epsa
                            ndpointer(shape=(3,3), dtype=float), #rb
                            ndpointer(shape=(3,3), dtype=float), #rp
                            ndpointer(shape=(3,3), dtype=float), #rbp
                            ndpointer(shape=(3,3), dtype=float), #rn
                            ndpointer(shape=(3,3), dtype=float)] #rbpn
def Pn00a(date1, date2):
    """ Precession-nutation, IAU 2000A model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 8-tuple:

            * nutation in longitude (float)
            * nutation in obliquity (float)
            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    *sofa manual.pdf page 162*
    """
    dpsi = c_double()
    deps = c_double()
    epsa = c_double()
    rb = asmatrix(zeros(shape=(3,3), dtype=float))
    rp = asmatrix(zeros(shape=(3,3), dtype=float))
    rbp = asmatrix(zeros(shape=(3,3), dtype=float))
    rn = asmatrix(zeros(shape=(3,3), dtype=float))
    rbpn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPn00a(date1, date2, byref(dpsi), byref(deps), byref(epsa), rb,
                    rp, rbp, rn, rbpn)
    return dpsi.value, deps.value, epsa.value, rb, rp, rbp, rn, rbpn


# iauPn00b
sofa.iauPn00b.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #dpsi
                            POINTER(c_double), #deps
                            POINTER(c_double), #epsa
                            ndpointer(shape=(3,3), dtype=float), #rb
                            ndpointer(shape=(3,3), dtype=float), #rp
                            ndpointer(shape=(3,3), dtype=float), #rbp
                            ndpointer(shape=(3,3), dtype=float), #rn
                            ndpointer(shape=(3,3), dtype=float)] #rbpn
def Pn00b(date1, date2):
    """ Precession-nutation, IAU 2000B model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 8-tuple:

            * nutation in longitude (float)
            * nutation in obliquity (float)
            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    *sofa manual.pdf page 164*
    """
    dpsi = c_double()
    deps = c_double()
    epsa = c_double()
    rb = asmatrix(zeros(shape=(3,3), dtype=float))
    rp = asmatrix(zeros(shape=(3,3), dtype=float))
    rbp = asmatrix(zeros(shape=(3,3), dtype=float))
    rn = asmatrix(zeros(shape=(3,3), dtype=float))
    rbpn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPn00b(date1, date2, byref(dpsi), byref(deps), byref(epsa), rb,
                    rp, rbp, rn, rbpn)
    return dpsi.value, deps.value, epsa.value, rb, rp, rbp, rn, rbpn


# iauPn06
sofa.iauPn06.argtypes = [c_double, #date1
                            c_double, #date2
                            c_double, #dpsi
                            c_double, #deps
                            POINTER(c_double), #epsa
                            ndpointer(shape=(3,3), dtype=float), #rb
                            ndpointer(shape=(3,3), dtype=float), #rp
                            ndpointer(shape=(3,3), dtype=float), #rbp
                            ndpointer(shape=(3,3), dtype=float), #rn
                            ndpointer(shape=(3,3), dtype=float)] #rbpn
def Pn06(date1, date2, dpsi, deps):
    """ Precession-nutation, IAU 2006 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param dpsi, deps: nutation.
    :type dpsi, deps: float

    :returns: a 6-tuple:

            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    *sofa manual.pdf page 166*
    """
    epsa = c_double()
    rb = asmatrix(zeros(shape=(3,3), dtype=float))
    rp = asmatrix(zeros(shape=(3,3), dtype=float))
    rbp = asmatrix(zeros(shape=(3,3), dtype=float))
    rn = asmatrix(zeros(shape=(3,3), dtype=float))
    rbpn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPn06(date1, date2, float(dpsi), float(deps), byref(epsa),
                                                        rb, rp, rbp, rn, rbpn)
    return epsa.value, rb, rp, rbp, rn, rbpn


# iauPn06a
sofa.iauPn06a.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #dpsi
                            POINTER(c_double), #deps
                            POINTER(c_double), #epsa
                            ndpointer(shape=(3,3), dtype=float), #rb
                            ndpointer(shape=(3,3), dtype=float), #rp
                            ndpointer(shape=(3,3), dtype=float), #rbp
                            ndpointer(shape=(3,3), dtype=float), #rn
                            ndpointer(shape=(3,3), dtype=float)] #rbpn
def Pn06a(date1, date2):
    """ Precession-nutation, IAU 2006/2000A models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 8-tuple:

            * nutation in longitude (float)
            * nutation in obliquity (float)
            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    *sofa manual.pdf page 168*
    """
    dpsi = c_double()
    deps = c_double()
    epsa = c_double()
    rb = asmatrix(zeros(shape=(3,3), dtype=float))
    rp = asmatrix(zeros(shape=(3,3), dtype=float))
    rbp = asmatrix(zeros(shape=(3,3), dtype=float))
    rn = asmatrix(zeros(shape=(3,3), dtype=float))
    rbpn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPn06a(date1, date2, byref(dpsi), byref(deps), byref(epsa), rb,
                    rp, rbp, rn, rbpn)
    return dpsi.value, deps.value, epsa.value, rb, rp, rbp, rn, rbpn


# iauPnm00a
sofa.iauPnm00a.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rbpn
def Pnm00a(date1, date2):
    """ Form the matrix of precession-nutation for a given date (including 
    frame bias), equinox-based, IAU 2000A model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: classical *NPB* matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 170*
    """
    rbpn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPnm00a(date1, date2, rbpn)
    return rbpn


# iauPnm00b
sofa.iauPnm00b.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rbpn
def Pnm00b(date1, date2):
    """ Form the matrix of precession-nutation for a given date (including 
    frame bias), equinox-based, IAU 2000B model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession-nutation matrix, as a numpy.matrix of shape \
        3x3.

    *sofa manual.pdf page 171*
    """
    rbpn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPnm00b(date1, date2, rbpn)
    return rbpn


# iauPnm06a
sofa.iauPnm06a.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rbpn
def Pnm06a(date1, date2):
    """ Form the matrix of precession-nutation for a given date (including 
    frame bias), IAU 2006 precession and IAU 2000A nutation models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession-nutation matrix, as a numpy.matrix of shape \
        3x3.

    *sofa manual.pdf page 172*
    """
    rbpn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPnm06a(date1, date2, rbpn)
    return rbpn


# iauPnm80
sofa.iauPnm80.argtypes = [c_double, #date1
                            c_double, #date2
                            ndpointer(shape=(3,3), dtype=float)] #rmatpn
def Pnm80(date1, date2):
    """ Form the matrix of precession/nutation for a given date, IAU 1976 
    precession model, IAU 1980 nutation model).

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: combined precessoin/nutation matrix, as a numpy.matrix of shape \
        3x3.

    *sofa manual.pdf page 173*
    """
    rmatpn = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPnm80(date1, date2, rmatpn)
    return rmatpn


# iauPom00
sofa.iauPom00.argtypes = [c_double, #xp
                            c_double, #yp
                            c_double, #sp
                            ndpointer(shape=(3,3), dtype=float)] #rpom
def Pom00(xp, yp, sp):
    """ Form the matrix of polar motion for a given date, IAU 2000.

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :param sp: the TIO locator in radians.
    :type sp: float

    :returns: the polar motion matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 174*
    """
    rpom = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauPom00(float(xp), float(yp), float(sp), rpom)
    return rpom


# iauPpp
sofa.iauPpp.argtypes = [ndpointer(shape=(1,3), dtype=float), #a
                        ndpointer(shape=(1,3), dtype=float), #b
                        ndpointer(shape=(1,3), dtype=float)] #apb
def Ppp(a, b):
    """ P-vector addition.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a + b as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 175*
    """
    apb = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauPpp(asmatrix(a, dtype=float), asmatrix(b, dtype=float), apb)
    return apb


# iauPpsp
sofa.iauPpsp.argtypes = [ndpointer(shape=(1,3), dtype=float), #a
                        c_double, #s
                        ndpointer(shape=(1,3), dtype=float), #b
                        ndpointer(shape=(1,3), dtype=float)] #apsb
def Ppsp(a, s, b):
    """ P-vector plus scaled p-vector.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param s: scalar (multiplier for *b*).
    :type s: float

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a + s*b as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 176*
    """
    apsb = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauPpsp(asmatrix(a, dtype=float), s, asmatrix(b, dtype=float), apsb)
    return apsb


# iauPr00
sofa.iauPr00.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #dpsipr
                            POINTER(c_double)] #depspr
def Pr00(date1, date2):
    """ Precession-rate part of the IAU 2000 precession-nutation models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * precession correction in longitude (float)
        * precession correction in obliquity (float).

    *sofa manual.pdf page 177*
    """
    dpsipr = c_double()
    depspr = c_double()
    sofa.iauPr00(date1, date2, byref(dpsipr), byref(depspr))
    return dpsipr.value, depspr.value


# iauPrec76
sofa.iauPrec76.argtypes = [c_double, #ep01
                            c_double, #ep02
                            c_double, #ep11
                            c_double, #ep12
                            POINTER(c_double), #zeta
                            POINTER(c_double), #z
                            POINTER(c_double)] #theta
def Prec76(ep01, ep02, ep11, ep12):
    """ Form the three Euler angles wich implement general precession between 
    two epochs, using IAU 1976 model (as for FK5 catalog).

    :param ep01, ep02: two-part TDB starting epoch.
    :type ep01, ep02: float

    :param ep11, ep12: two-part TDB ending epoch.
    :type ep11, ep12: float

    :returns: a 3-tuple:

            * 1st rotation: radians cw around z (float)
            * 3rd rotation: radians cw around z (float)
            * 2nd rotation: radians ccw around y (float).

    *sofa manual.pdf page 179*
    """
    zeta = c_double()
    z = c_double()
    theta = c_double()
    sofa.iauPrec76(ep01, ep02, ep11, ep12, byref(zeta), byref(z), byref(theta))
    return zeta.value, z.value, theta.value


# iauPv2p
sofa.iauPv2p.argtypes = [ndpointer(shape=(2,3), dtype=float), #pv
                            ndpointer(shape=(1,3), dtype=float)] #p
def Pv2p(pv):
    """ Discard velocity component of a pv-vector.

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: p-vector as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 181*
    """
    p = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauPv2p(asmatrix(pv, dtype=float), p)
    return p


# iauPv2s
sofa.iauPv2s.argtypes = [ndpointer(shape=(2,3), dtype=float), #pv
                            POINTER(c_double), #theta
                            POINTER(c_double), #phi
                            POINTER(c_double), #r
                            POINTER(c_double), #td
                            POINTER(c_double), #pd
                            POINTER(c_double)] #rd
def Pv2s(pv):
    """ Convert position/velocity from cartesian to spherical coordinates.

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a 6-tuple:

        * longitude angle :math:`\\theta`  in radians (float)
        * latitude angle :math:`\phi` in radians (float)
        * radial distance *r* (float)
        * rate of change of :math:`\\theta` (float)
        * rate of change of :math:`\phi` (float)
        * rate of change of *r* (float)

    *sofa manual.pdf page 182*
    """
    theta = c_double()
    phi = c_double()
    r = c_double()
    td = c_double()
    pd = c_double()
    rd = c_double()
    sofa.iauPv2s(asmatrix(pv, dtype=float), byref(theta), byref(phi), byref(r),
                    byref(td), byref(pd), byref(rd))
    return theta.value, phi.value, r.value, td.value, pd.value, rd.value


# iauPvdpv
sofa.iauPvdpv.argtypes = [ndpointer(shape=(2,3), dtype=float), #a
                            ndpointer(shape=(2,3), dtype=float), #b
                            ndpointer(shape=(1,2), dtype=float)] #adb
def Pvdpv(a, b):
    """ Inner product of two pv-vectors.

    :param a: first pv-vector.
    :type a: array-like of shape (2,3)

    :param b: second pv-vector.
    :type b: array-like of shape (2,3)

    :returns: a . b as a numpy.matrix of shape 1x2.

    *sofa manual.pdf page 183*
    """
    adb = asmatrix(zeros(shape=(2), dtype=float))
    sofa.iauPvdpv(asmatrix(a, dtype=float), asmatrix(b, dtype=float), adb)
    return adb


# iauPvm
sofa.iauPvm.argtypes = [ndpointer(shape=(2,3), dtype=float), #pv
                        POINTER(c_double), #r
                        POINTER(c_double)] #s
def Pvm(pv):
    """ Modulus of pv-vector.

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a 2-tuple:

        * modulus of position component (float)
        * modulus of velocity component (float).

    *sofa manual.pdf page 184*
    """
    r = c_double()
    s = c_double()
    sofa.iauPvm(asmatrix(pv, dtype=float), byref(r), byref(s))
    return r.value, s.value


# iauPvmpv
sofa.iauPvmpv.argtypes = [ndpointer(shape=(2,3), dtype=float), #a
                            ndpointer(shape=(2,3), dtype=float), #b
                            ndpointer(shape=(2,3), dtype=float)] #amb
def Pvmpv(a, b):
    """ Subtract one pv-vector from another.

    :param a: first pv-vector.
    :type a: array-like of shape (2,3)

    :param b: second pv-vector.
    :type b: array-like of shape (2,3)

    :returns: a - b as a numpy.matrix of shape 2x3.

    *sofa manual.pdf page 185*
    """
    amb = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauPvmpv(asmatrix(a, dtype=float), asmatrix(b, dtype=float), amb)
    return amb


# iauPvppv
sofa.iauPvppv.argtypes = [ndpointer(shape=(2,3), dtype=float), #a
                            ndpointer(shape=(2,3), dtype=float), #b
                            ndpointer(shape=(2,3), dtype=float)] #apb
def Pvppv(a, b):
    """ Add one pv-vector to another.

    :param a: first pv-vector.
    :type a: array-like of shape (2,3)

    :param b: second pv-vector.
    :type b: array-like of shape (2,3)

    :returns: a + b as a numpy.matrix of shape 2x3.

    *sofa manual.pdf page 186*
    """
    apb = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauPvppv(asmatrix(a, dtype=float), asmatrix(b, dtype=float), apb)
    return apb


# iauPvstar
sofa.iauPvstar.argtypes = [ndpointer(shape=(2,3), dtype=float), #pv
                            POINTER(c_double), #ra
                            POINTER(c_double), #dec
                            POINTER(c_double), #pmr
                            POINTER(c_double), #pmd
                            POINTER(c_double), #px
                            POINTER(c_double)] #rv
sofa.iauPvstar.restype = c_int
def Pvstar(pv):
    """ Convert star position-velocity vector to catalog coordinates.

    :param pv: pv-vector (AU, AU/day).
    :type pv: array-like of shape (2,3)

    :returns: a 7-tuple:

        * right ascensin in radians (float)
        * declination in radians (float)
        * RA proper motion (radians/year) (float)
        * Dec proper motion (radians/year) (float)
        * parallax in arcseconds (float)
        * radial velocity (km/s, positive = receding)
        * function status.

    *sofa manual.pdf page 187*
    """
    ra = c_double()
    dec = c_double()
    pmr = c_double()
    pmd = c_double()
    px = c_double()
    rv = c_double()
    status = sofa.iauPvstar(asmatrix(pv, dtype=float), byref(ra), byref(dec),
                                byref(pmr), byref(pmd), byref(px), byref(rv))
    return ra.value, dec.value, pmr.value, pmd.value, px.value, rv.value, status


# iauPvu
sofa.iauPvu.argtypes = [c_double, #dt
                        ndpointer(shape=(2,3), dtype=float), #pv
                        ndpointer(shape=(2,3), dtype=float)] #upv
def Pvu(dt, pv):
    """ Update a pv-vector.

    :param dt: time interval.
    :type dt: float

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a new pv-vector as a numpy.matrix of shape 2x3, with p \
        updated and v unchanged.

    *sofa manual.pdf page 189*
    """
    upv = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauPvu(dt, asmatrix(pv, dtype=float), upv)
    return upv


# iauPvup
sofa.iauPvup.argtypes = [c_double, #dt
                            ndpointer(shape=(2,3), dtype=float), #pv
                            ndpointer(shape=(1,3), dtype=float)] #p
def Pvup(dt, pv):
    """ Update a pv-vector, discarding the velocity component.

    :param dt: time interval.
    :type dt: float

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a new p-vector, as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 190*
    """
    p = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauPvup(dt, asmatrix(pv, dtype=float), p)
    return p


# iauPvxpv
sofa.iauPvxpv.argtypes = [ndpointer(shape=(2,3), dtype=float), #a
                            ndpointer(shape=(2,3), dtype=float), #b
                            ndpointer(shape=(2,3), dtype=float)] #axb
def Pvxpv(a, b):
    """ Outer product of two pv-vectors.

    :param a: first pv-vector.
    :type a: array-like of shape (2,3)

    :param b: second pv-vector.
    :type b: array-like of shape (2,3)

    :returns: a x b as a numpy.matrix of shape 2x3.

    *sofa manual.pdf page 191*
    """
    axb = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauPvxpv(asmatrix(a, dtype=float), asmatrix(b, dtype=float), axb)
    return axb


# iauPxp
sofa.iauPxp.argtypes = [ndpointer(shape=(1,3), dtype=float), #a
                        ndpointer(shape=(1,3), dtype=float), #b
                        ndpointer(shape=(1,3), dtype=float)] #axb
def Pxp(a, b):
    """ P-vector outer product.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a x b as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 192*
    """
    axb = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauPxp(asmatrix(a, dtype=float), asmatrix(b, dtype=float), axb)
    return axb


# iauRm2v
sofa.iauRm2v.argtypes = [ndpointer(shape=(3,3), dtype=float), #r
                        ndpointer(shape=(1,3), dtype=float)] #w
def Rm2v(r):
    """ Express a r-matrix as a r-vector.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: rotation vector as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 193*
    """
    w = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauRm2v(asmatrix(r, dtype=float), w)
    return w


# iauRv2m
sofa.iauRv2m.argtypes = [ndpointer(shape=(1,3), dtype=float), #w
                        ndpointer(shape=(3,3), dtype=float)] #r
def Rv2m(w):
    """ Form the rotation matrix corresponding to a given r-vector.

    :param w: rotation vector.
    :type w: array-like of shape (1,3)

    :returns: rotation matrix as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 194*
    """
    r = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauRv2m(asmatrix(w, dtype=float), r)
    return r


# iauRx
sofa.iauRx.argtypes = [c_double, #phi
                        ndpointer(shape=(3,3), dtype=float)] #r
def Rx(phi, r):
    """ Rotate a r-matrix about the x-axis.

    :param phi: angle in radians.
    :type phi: float

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: the new rotation matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 195*
    """
    r2 = asmatrix(r, dtype=float).copy()
    sofa.iauRx(float(phi), r2)
    return r2


# iauRxp
sofa.iauRxp.argtypes = [ndpointer(shape=(3,3), dtype=float), #r
                        ndpointer(shape=(1,3), dtype=float), #p
                        ndpointer(shape=(1,3), dtype=float)] #rp
def Rxp(r, p):
    """ Multiply a p-vector by a r-matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: r * p as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 196*
    """
    rp = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauRxp(asmatrix(r, dtype=float), asmatrix(p, dtype=float), rp)
    return rp


# iauRxpv
sofa.iauRxpv.argtypes = [ndpointer(shape=(3,3), dtype=float), #r
                        ndpointer(shape=(2,3), dtype=float), #pv
                        ndpointer(shape=(2,3), dtype=float)] #rpv
def Rxpv(r, pv):
    """ Multiply a pv-vector by a r-matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: r * pv as a numpy.matrix of shape 2x3.

    *sofa manual.pdf page 197*
    """
    rpv = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauRxpv(asmatrix(r, dtype=float), asmatrix(pv, dtype=float), rpv)
    return rpv


# iauRxr
sofa.iauRxr.argtypes = [ndpointer(shape=(3,3), dtype=float), #a
                        ndpointer(shape=(3,3), dtype=float), #b
                        ndpointer(shape=(3,3), dtype=float)] #atb
def Rxr(a, b):
    """ Multiply two rotation matrices.

    :param a: first r-matrix.
    :type a: array-like of shape (3,3)

    :param b: second r-matrix.
    :type b: array-like of shape (3,3)

    :returns: a * b as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 198*
    """
    atb = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauRxr(asmatrix(a, dtype=float), asmatrix(b, dtype=float), atb)
    return atb


# iauRy
sofa.iauRy.argtypes = [c_double, #theta
                        ndpointer(shape=(3,3), dtype=float)] #r
def Ry(theta, r):
    """ Rotate a r-matrix about the y-axis.

    :param theta: angle in radians.
    :type theta: float

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: the new rotation matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 199*
    """
    r2 = asmatrix(r).copy()
    sofa.iauRy(float(theta), r2)
    return r2


# iauRz
sofa.iauRz.argtypes = [c_double, #psi
                        ndpointer(shape=(3,3), dtype=float)] #r
def Rz(psi, r):
    """ Rotate a r-matrix about the z-axis.

    :param psi: angle in radians.
    :type psi: float

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: the new rotation matrix, as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 200*
    """
    r2 = asmatrix(r).copy()
    sofa.iauRz(float(psi), r2)
    return r2


# iauS00
sofa.iauS00.argtypes = [c_double, #date1
                        c_double, #date2
                        c_double, #x
                        c_double] #y
sofa.iauS00.restype = c_double
def S00(date1, date2, x, y):
    """ The CIO locator *s*, positioning the celestial intermediate 
    origin on the equator of the celestial intermediate pole, given the
    CIP's X,Y coordinates. Compatible with IAU 2000A precession-nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param x, y: CIP coordinates.
    :type x, y: float

    :returns: the CIO locator *s* in radians (float).

    *sofa manual.pdf page 201*
    """
    return sofa.iauS00(date1, date2, float(x), float(y))


# iauS00a
sofa.iauS00a.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauS00a.restype = c_double
def S00a(date1, date2):
    """ The CIO locator, positioning the celestial intermediate origin 
    on the equator of the celestial intermediate pole, using IAU 2000A
    precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the CIO locator *s* in radians (float):

    *sofa manual.pdf page 203*
    """
    return sofa.iauS00a(date1, date2)


# iauS00b
sofa.iauS00b.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauS00b.restype = c_double
def S00b(date1, date2):
    """ The CIO locator, positioning the celestial intermediate origin 
    on the equator of the celestial intermediate pole, using IAU 2000B
    precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the CIO locator *s* in radians (float):

    *sofa manual.pdf page 205*
    """
    return sofa.iauS00b(date1, date2)


# iauS06
sofa.iauS06.argtypes = [c_double, #date1
                        c_double, #date2
                        c_double, #x
                        c_double] #y
sofa.iauS06.restype = c_double
def S06(date1, date2, x, y):
    """ The CIO locator *s*, positioning the celestial intermediate 
    origin on the equator of the celestial intermediate pole, given the
    CIP's X,Y coordinates. Compatible with IAU 2006/2000A precession-nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param x, y: CIP coordinates.
    :type x, y: float

    :returns: the CIO locator *s* in radians (float).

    *sofa manual.pdf page 207*
    """
    return sofa.iauS06(date1, date2, float(x), float(y))


# iauS06a
sofa.iauS06a.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauS06a.restype = c_double
def S06a(date1, date2):
    """ The CIO locator, positioning the celestial intermediate origin 
    on the equator of the celestial intermediate pole, using IAU 2006
    precession and IAU 2000A nutation models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the CIO locator *s* in radians (float):

    *sofa manual.pdf page 209*
    """
    return sofa.iauS06a(date1, date2)


# iauS2c
sofa.iauS2c.argtypes = [c_double, #theta
                        c_double, #phi
                        ndpointer(shape=(1,3), dtype=float)] #c
def S2c(theta, phi):
    """ Convert spherical coordinates to cartesian.

    :param theta: longitude angle in radians.
    :type theta: float

    :param phi: latitude angle in radians.
    :type phi: float

    :returns: direction cosines as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 211*
    """
    c = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauS2c(float(theta), float(phi), c)
    return c


# iauS2p
sofa.iauS2p.argtypes = [c_double, #theta
                        c_double, #phi
                        c_double, #r
                        ndpointer(shape=(1,3), dtype=float)] #p
def S2p(theta, phi, r):
    """ Convert spherical polar coordinates to p-vector.

    :param theta: longitude angle in radians.
    :type theta: float

    :param phi: latitude angle in radians.
    :type phi: float

    :param r: radial distance.
    :type r: float

    :returns: cartesian coordinates as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 212*
    """
    p = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauS2p(float(theta), float(phi), r, p)
    return p


# iauS2pv
sofa.iauS2pv.argtypes = [c_double, #theta
                            c_double, #phi
                            c_double, #r
                            c_double, #td
                            c_double, #pd
                            c_double, #rd
                            ndpointer(shape=(2,3), dtype=float)] #pv
def S2pv(theta, phi, r, td, pd, rd):
    """ Convert position/velocity from spherical to cartesian coordinates.

    :param theta: longitude angle in radians.
    :type theta: float

    :param phi: latitude angle in radians.
    :type phi: float

    :param r: radial distance.
    :type r: float

    :param td: rate of change of *theta*.
    :type td: float

    :param pd: rate of change of *phi*.
    :type pd: float

    :param rd: rate of change of *r*.
    :type rd: float

    :returns: pv-vector as a numpy.matrix of shape 2x3.

    *sofa manual.pdf page 213*
    """
    pv = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauS2pv(float(theta), float(phi), r, float(td), float(pd), rd, pv)
    return pv


# iauS2xpv
sofa.iauS2xpv.argtypes = [c_double, #s1
                            c_double, #s2
                            ndpointer(shape=(2,3), dtype=float), #pv
                            ndpointer(shape=(2,3), dtype=float)] #spv
def S2xpv(s1, s2, pv):
    """ Multiply a pv-vector by two scalars.

    :param s1: scalar to multiply position component by.
    :type s1: float

    :param s2; scalar to multiply velocity component by.
    :type s2: float

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a new pv-vector (with p scaled by s1 and v scaled by s2) as a \
        numpy.matrix of shape 2x3.

    *sofa manual.pdf page 214*
    """
    spv = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauS2xpv(s1, s2, asmatrix(pv, dtype=float), spv)
    return spv


# iauSepp
sofa.iauSepp.argtypes = [ndpointer(shape=(1,3), dtype=float), #a
                            ndpointer(shape=(1,3), dtype=float)] #b
sofa.iauSepp.restype = c_double
def Sepp(a, b):
    """ Angular separation between two p-vectors.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: angular separation in radians, always positive (float).

    *sofa manual.pdf page 215*
    """
    return sofa.iauSepp(asmatrix(a, dtype=float), asmatrix(b, dtype=float))


# iauSeps
sofa.iauSeps.argtypes = [c_double, #al
                            c_double, #ap
                            c_double, #bl
                            c_double] #bp
sofa.iauSeps.restype = c_double
def Seps(al, ap, bl, bp):
    """ Angular separation between two sets of spherical coordinates.

    :param al: first longitude in radians.
    :type al: float

    :param ap: first latitude in radians.
    :type ap: float

    :param bl: second longitude in radians.
    :type bl: float

    :param bl: second latitude in radians.
    :type bp: float

    :returns: angular separation in radians (float).

    *sofa manual.pdf page 216*
    """
    return sofa.iauSeps(float(al), float(ap), float(bl), float(bp))


# iauSp00
sofa.iauSp00.argtypes = [c_double, #date1
                            c_double] #date2
sofa.iauSp00.restype = c_double
def Sp00(date1, date2):
    """ The TIO locator, positioning the terrestrial intermediate origin on 
    the equator of the celestial intermediate pole.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the TIO locator in radians (float).

    *sofa manual.pdf page 217*
    """
    return sofa.iauSp00(date1, date2)


# iauStarpm
sofa.iauStarpm.argtypes = [c_double, #ra1
                            c_double, #dec1
                            c_double, #pmr1
                            c_double, #pmd1
                            c_double, #px1
                            c_double, #rv1
                            c_double, #ep1a
                            c_double, #ep1b
                            c_double, #ep2a
                            c_double, #ep2b
                            POINTER(c_double), #ra2
                            POINTER(c_double), #dec2
                            POINTER(c_double), #pmr2
                            POINTER(c_double), #pmd2
                            POINTER(c_double), #px2
                            POINTER(c_double)] #rv2
sofa.iauStarpm.restype = c_int
def Starpm(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    """ Update star catalog data for space motion.

    :param ra1: right ascension in radians.
    :type ra1: float

    :param dec1: declination in radians.
    :type dec1: float

    :param pmr1: proper motion in RA (radians/year).
    :type pmr1: float

    :param pmd1: proper motion in Dec (radians/year).
    :type pmd1: float

    :param px1: parallax in arcseconds.
    :type px1: float

    :param rv1: radial velocity (km/s, positive = receding).
    :type rv1: float

    :param ep1a, ep1b: two-part starting epoch.
    :type ep1a, ep1b: float

    :param ep2a, ep2b: two-part ending epoch.
    :type ep2a, ep2b: float

    :returns: a 7-tuple:

        * the new right ascension in radians (float)
        * the new declination in radians (float)
        * the new RA proper motion in radians/year (float)
        * the new Dec proper motion in radians/year (float)
        * the new parallax in arcseconds (float)
        * the new radial velocity (km/s)
        * function status.

    *sofa manual.pdf page 218*
    """
    ra2 = c_double()
    dec2 = c_double()
    pmr2 = c_double()
    pmd2 = c_double()
    px2 = c_double()
    rv2 = c_double()
    status = sofa.iauStarpm(float(ra1), float(dec1), float(pmr1), float(pmd1),
                                float(px1), float(rv1), ep1a, ep1b, ep2a,
                    ep2b, byref(ra2), byref(dec2), byref(pmr2), byref(pmd2),
                    byref(px2), byref(rv2))
    return ra2.value, dec2.value, pmr2.value, pmd2.value, px2.value, rv2.value,\
            status


# iauStarpv
sofa.iauStarpv.argtypes = [c_double, #ra
                            c_double, #dec
                            c_double, #pmr
                            c_double, #pmd
                            c_double, #px
                            c_double, #rv
                            ndpointer(shape=(2,3), dtype=float)] #pv
sofa.iauStarpv.restype = c_int
def Starpv(ra, dec, pmr, pmd, px, rv):
    """ Convert star catalog coordinates to position+velocity vector.

    :param ra: right ascension in radians.
    :type ra: float

    :param dec: declination in radians.
    :type dec: float

    :param pmr: proper motion in RA (radians/year).
    :type pmr: float

    :param pmd: proper motion in Dec (radians/year).
    :type pmd: float

    :param px: parallax in arcseconds.
    :type px: float

    :param rv: radial velocity (km/s, positive = receding).
    :type rv: float

    :returns: a 2-tuple:

        * pv-vector (AU, AU/day) as a numpy.matrix of shape 2x3
        * function status.

    *sofa manual.pdf page 220*
    """
    pv = asmatrix(zeros(shape=(2,3), dtype=float))
    status = sofa.iauStarpv(float(ra), float(dec), float(pmr), float(pmd),
                                                    float(px), float(rv), pv)
    return pv, status


# iauSxp
sofa.iauSxp.argtypes = [c_double, #s
                        ndpointer(shape=(1,3), dtype=float), #p
                        ndpointer(shape=(1,3), dtype=float)] #sp
def Sxp(s, p):
    """ Multiply a p-vector by a scalar.

    :param s: scalar.
    :type s: float

    :param p: p-vector
    :type p: array-like of shape (1,3)

    :returns: s * p as a numpy.matrix of shape 1x3.

    *sofa manual.pdf page 222*
    """
    sp = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauSxp(s, asmatrix(p, dtype=float), sp)
    return sp


# iauSxpv
sofa.iauSxpv.argtypes = [c_double, #s
                            ndpointer(shape=(2,3), dtype=float), #pv
                            ndpointer(shape=(2,3), dtype=float)] #spv
def Sxpv(s, pv):
    """ Multiply a pv-vector by a scalar.

    :param s: scalar.
    :type s: float

    :param pv: pv-vector
    :type pv: array-like of shape (2,3)

    :returns: s * pv as a numpy.matrix of shape 2x3.

    *sofa manual.pdf page 223*
    """
    spv = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauSxpv(s, asmatrix(pv, dtype=float), spv)
    return spv


# iauTaitt
sofa.iauTaitt.argtypes = [c_double, #tai1
                            c_double, #tai2
                            POINTER(c_double), #tt1
                            POINTER(c_double)] #tt2
sofa.iauTaitt.restype = c_int
def Taitt(tai1, tai2):
    """ Timescale transformation: International Atomic Time (TAI) to
    Terrestrial Time (TT).

    :param tai1, tai2: TAI as a two-part Julian Date.
    :type tai1, tai2: float

    :returns: a tuple of three items:

        * TT as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 224*
    """
    tt1 = c_double()
    tt2 = c_double()
    s = sofa.iauTaitt(tai1, tai2, byref(tt1), byref(tt2))
    return tt1.value, tt2.value, s


# iauTaiut1
sofa.iauTaiut1.argtypes = [c_double, #tai1
                            c_double, #tai2
                            c_double, #dta
                            POINTER(c_double), #ut11
                            POINTER(c_double)] #ut12
sofa.iauTaiut1.restype = c_int
def Taiut1(tai1, tai2, dta):
    """ Timescale transformation: International Atomic Time (TAI) to
    Universal Time (UT1).

    :param tai1, tai2: TAI as a two-part Julian Date.
    :type tai1, tai2: float

    :param dta: UT1-TAI in seconds.
    :type dta: float

    :returns: a tuple of three items:

        * UT1 as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 225*
    """
    ut11 = c_double()
    ut12 = c_double()
    s = sofa.iauTaiut1(tai1, tai2, dta, byref(ut11), byref(ut12))
    return ut11.value, ut12.value, s


# iauTaiutc
sofa.iauTaiutc.argtypes = [c_double, #tai1
                            c_double, #tai2
                            POINTER(c_double), #utc1
                            POINTER(c_double)] #utc2
sofa.iauTaiutc.restype = c_int
def Taiutc(tai1, tai2):
    """ Timescale transformation: International Atomic Time (TAI) to
    Terrestrial Time (TT).

    :param tai1, tai2: TAI as a two-part Julian Date.
    :type tai1, tai2: float

    :returns: a tuple of three items:

        * UTC as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 226*
    """
    utc1 = c_double()
    utc2 = c_double()
    s = sofa.iauTaiutc(tai1, tai2, byref(utc1), byref(utc2))
    return utc1.value, utc2.value, s


# iauTcbtdb
sofa.iauTcbtdb.argtypes = [c_double, #tcb1
                            c_double, #tcb2
                            POINTER(c_double), #tdb1
                            POINTER(c_double)] #tdb2
sofa.iauTcbtdb.restype = c_int
def Tcbtdb(tcb1, tcb2):
    """ Timescale transformation: Barycentric Coordinate Time (TCB) to
    Barycentric dynamical time (TDB).

    :param tcb1, tcb2: TCB as a two-part Julian Date.
    :type tcb1, tcb2: float

    :returns: a tuple of three items:

        * TDB as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 227*
    """
    tdb1 = c_double()
    tdb2 = c_double()
    s = sofa.iauTcbtdb(tcb1, tcb2, byref(tdb1), byref(tdb2))
    return tdb1.value, tdb2.value, s


# iauTcgtt
sofa.iauTcgtt.argtypes = [c_double, #tcg1
                            c_double, #tcg2
                            POINTER(c_double), #tt1
                            POINTER(c_double)] #tt2
sofa.iauTcgtt.restype = c_int
def Tcgtt(tcg1, tcg2):
    """ Timescale transformation: Geocentric Coordinate Time (TCG) to
    Terrestrial Time (TT).

    :param tcg1, tcg2: TCG as a two-part Julian Date.
    :type tcg1, tcg2: float

    :returns: a tuple of three items:

        * TT as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 228*
    """
    tt1 = c_double()
    tt2 = c_double()
    s = sofa.iauTcgtt(tcg1, tcg2, byref(tt1), byref(tt2))
    return tt1.value, tt2.value, s


# iauTdbtcb
sofa.iauTdbtcb.argtypes = [c_double, #tdb1
                            c_double, #tdb2
                            POINTER(c_double), #tcb1
                            POINTER(c_double)] #tcb2
sofa.iauTdbtcb.restype = c_int
def Tdbtcb(tdb1, tdb2):
    """ Timescale transformation: Barycentric Dynamical Time (TDB) to
    Barycentric Coordinate Time (TCB).

    :param tdb1, tdb2: TDB as a two-part Julian Date.
    :type tdb1, tdb2: float

    :returns: a tuple of three items:

        * TCB as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 229*
    """
    tcb1 = c_double()
    tcb2 = c_double()
    s = sofa.iauTdbtcb(tdb1, tdb2, byref(tcb1), byref(tcb2))
    return tcb1.value, tcb2.value, s


# iauTdbtt
sofa.iauTdbtt.argtypes = [c_double, #tdb1
                            c_double, #tdb2
                            c_double, #dtr
                            POINTER(c_double), #tt1
                            POINTER(c_double)] #tt2
sofa.iauTdbtt.restype = c_int
def Tdbtt(tdb1, tdb2, dtr):
    """ Timescale transformation: Barycentric Dynamical Time (TDB) to
    Terrestrial Time (TT).

    :param tdb1, tdb2: TDB as a two-part Julian Date.
    :type tdb1, tdb2: float

    :param dtr: TDB-TT in seconds.
    :type dtr: float

    :returns: a tuple of three items:

        * TT as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 230*
    """
    tt1 = c_double()
    tt2 = c_double()
    s = sofa.iauTdbtt(tdb1, tdb2, dtr, byref(tt1), byref(tt2))
    return tt1.value, tt2.value, s


# iauTf2a
sofa.iauTf2a.argtypes = [c_char, #s
                            c_int, #ihour
                            c_int, #imin
                            c_double, #sec
                            POINTER(c_double)] #rad
sofa.iauTf2a.restype = c_int
def Tf2a(s, ihour, imin, sec):
    """ Convert hours, minutes, seconds to radians.

    :param s: sign, '-' is negative, everything else positive.

    :param ihour: hours.
    :type ihour: int

    :param imin: minutes.
    :type imin: int

    :param sec: seconds.
    :type sec: float

    :returns: a tuple of two items:

        * the converted value in radians (float)
        * function status.

    *sofa manual.pdf page 231*
    """

    rad = c_double()
    s = sofa.iauTf2a(str(s), ihour, imin, sec, byref(rad))
    return rad.value, s


# iauTf2d
sofa.iauTf2d.argtypes = [c_char, #s
                            c_int, #ihour
                            c_int, #imin
                            c_double, #sec
                            POINTER(c_double)] #days
sofa.iauTf2d.restype = c_int
def Tf2d(s, ihour, imin, sec):
    """ Convert hours, minutes, seconds to days.

    :param s: sign, '-' is negative, everything else positive.

    :param ihour: hours.
    :type ihour: int

    :param imin: minutes.
    :type imin: int

    :param sec: seconds.
    :type sec: float

    :returns: a tuple of two items:

        * the converted value in days (float)
        * function status.

    *sofa manual.pdf page 232*
    """

    days = c_double()
    s = sofa.iauTf2d(str(s), ihour, imin, sec, byref(days))
    return days.value, s


# iauTr
sofa.iauTr.argtypes = [ndpointer(shape=(3,3), dtype=float), #r
                        ndpointer(shape=(3,3), dtype=float)] #rt
def Tr(r):
    """ Transpose a rotation matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: transpose as a numpy.matrix of shape 3x3.

    *sofa manual.pdf page 233*
    """
    rt = asmatrix(zeros(shape=(3,3), dtype=float))
    sofa.iauTr(asmatrix(r, dtype=float), rt)
    return rt


# iauTrxp
sofa.iauTrxp.argtypes = [ndpointer(shape=(3,3), dtype=float), #r
                        ndpointer(shape=(1,3), dtype=float), #p
                        ndpointer(shape=(1,3), dtype=float)] #trp
def Trxp(r, p):
    """ Multiply a p-vector by the transpose of a rotation matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: numpy.matrix of shape 1x3.

    *sofa manual.pdf page 234*
    """
    trp = asmatrix(zeros(shape=(1,3), dtype=float))
    sofa.iauTrxp(asmatrix(r, dtype=float), asmatrix(p, dtype=float), trp)
    return trp


# iauTrxpv
sofa.iauTrxpv.argtypes = [ndpointer(shape=(3,3), dtype=float), #r
                            ndpointer(shape=(2,3), dtype=float), #pv
                            ndpointer(shape=(2,3), dtype=float)] #trpv
def Trxpv(r, pv):
    """ Multiply a pv-vector by the transpose of a rotation matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: numpy.matrix of shape 2x3.

    *sofa manual.pdf page 235*
    """
    trpv = asmatrix(zeros(shape=(2,3), dtype=float))
    sofa.iauTrxpv(asmatrix(r, dtype=float), asmatrix(pv, dtype=float), trpv)
    return trpv


# iauTttai
sofa.iauTttai.argtypes = [c_double, #tt1
                            c_double, #tt2
                            POINTER(c_double), #tai1
                            POINTER(c_double)] #tai2
sofa.iauTttai.restype = c_int
def Tttai(tt1, tt2):
    """ Timescale transformation: Terrestrial Time (TT) to
    International Atomic Time (TAI).

    :param tt1, tt2: TT as a two-part Julian Date.
    :type tt1, tt2: float

    :returns: a tuple of three items:

        * TAI as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 236*
    """
    tai1 = c_double()
    tai2 = c_double()
    s = sofa.iauTttai(tt1, tt2, byref(tai1), byref(tai2))
    return tai1.value, tai2.value, s


# iauTttcg
sofa.iauTttcg.argtypes = [c_double, #tt1
                            c_double, #tt2
                            POINTER(c_double), #tcg1
                            POINTER(c_double)] #tcg2
sofa.iauTttcg.restype = c_int
def Tttcg(tt1, tt2):
    """ Timescale transformation: Terrestrial Time (TT) to
    Geocentric Coordinate Time (TCG).

    :param tt1, tt2: TT as a two-part Julian Date.
    :type tt1, tt2: float

    :returns: a tuple of three items:

        * TCG as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 237*
    """
    tcg1 = c_double()
    tcg2 = c_double()
    s = sofa.iauTttcg(tt1, tt2, byref(tcg1), byref(tcg2))
    return tcg1.value, tcg2.value, s


# iauTttdb
sofa.iauTttdb.argtypes = [c_double, #tt1
                            c_double, #tt2
                            c_double, #dtr
                            POINTER(c_double), #tdb1
                            POINTER(c_double)] #tdb2
sofa.iauTttdb.restype = c_int
def Tttdb(tt1, tt2, dtr):
    """ Timescale transformation: Terrestrial Time (TT) to
    Barycentric Dynamical Time (TDB)

    :param tt1, tt2: TT as a two-part Julian Date.
    :type tt1, tt2: float

    :param dtr: TDB-TT in seconds.
    :type dtr: float

    :returns: a tuple of three items:

        * TDB as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 238*
    """
    tdb1 = c_double()
    tdb2 = c_double()
    s = sofa.iauTttdb(tt1, tt2, dtr, byref(tdb1), byref(tdb2))
    return tdb1.value, tdb2.value, s


# iauTtut1
sofa.iauTtut1.argtypes = [c_double, #tt1
                            c_double, #tt2
                            c_double, #dt
                            POINTER(c_double), #ut11
                            POINTER(c_double)] #ut12
sofa.iauTtut1.restype = c_int
def Ttut1(tt1, tt2, dt):
    """ Timescale transformation: Terrestrial Time (TT) to
    Universal Time (UT1).

    :param tt1, tt2: TT as a two-part Julian Date.
    :type tt1, tt2: float

    :param dt: TT-UT1 in seconds.
    :type dt: float

    :returns: a tuple of three items:

        * UT1 as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 239*
    """
    ut11 = c_double()
    ut12 = c_double()
    s = sofa.iauTtut1(tt1, tt2, dt, byref(ut11), byref(ut12))
    return ut11.value, ut12.value, s


# iauUt1tai
sofa.iauUt1tai.argtypes = [c_double, #ut11
                            c_double, #ut12
                            c_double, #dta
                            POINTER(c_double), #tai1
                            POINTER(c_double)] #tai2
sofa.iauUt1tai.restype = c_int
def Ut1tai(ut11, ut12, dta):
    """ Timescale transformation: Universal Time (UT1) to
    International Atomic Time (TAI).

    :param ut11, ut12: UT1 as a two-part Julian Date.
    :type ut11, ut12: float

    :param dta: UT1-TAI in seconds.
    :type dta: float

    :returns: a tuple of three items:

        * TAI as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 240*
    """
    tai1 = c_double()
    tai2 = c_double()
    s = sofa.iauUt1tai(ut11, ut12, dta, byref(tai1), byref(tai2))
    return tai1.value, tai2.value, s


# iauUt1tt
sofa.iauUt1tt.argtypes = [c_double, #ut11
                            c_double, #ut12
                            c_double, #dt
                            POINTER(c_double), #tt1
                            POINTER(c_double)] #tt2
sofa.iauUt1tt.restype = c_int
def Ut1tt(ut11, ut12, dt):
    """ Timescale transformation: Universal Time (UT1) to
    Terrestrial Time (TT).

    :param ut11, ut12: UT1 as a two-part Julian Date.
    :type ut11, ut12: float

    :param dt: TT-UT1 in seconds.
    :type dt: float

    :returns: a tuple of three items:

        * TT as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 241*
    """
    tt1 = c_double()
    tt2 = c_double()
    s = sofa.iauUt1tt(ut11, ut12, dt, byref(tt1), byref(tt2))
    return tt1.value, tt2.value, s


# iauUt1utc
sofa.iauUt1utc.argtypes = [c_double, #ut11
                            c_double, #ut12
                            c_double, #dut1
                            POINTER(c_double), #utc1
                            POINTER(c_double)] #utc2
sofa.iauUt1utc.restype = c_int
def Ut1utc(ut11, ut12, dut1):
    """ Timescale transformation: Universal Time (UT1) to
    Coordinated Universal Time (UTC)

    :param ut11, ut12: UT1 as a two-part Julian Date.
    :type ut11, ut12: float

    :param dut1: UT1-UTC in seconds.
    :type dut1: float

    :returns: a tuple of three items:

        * UTC as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 242*
    """
    utc1 = c_double()
    utc2 = c_double()
    s = sofa.iauUt1utc(ut11, ut12, dut1, byref(utc1), byref(utc2))
    return utc1.value, utc2.value, s


# iauUtctai
sofa.iauUtctai.argtypes = [c_double, #utc1
                            c_double, #utc2
                            POINTER(c_double), #tai1
                            POINTER(c_double)] #tai2
sofa.iauUtctai.restype = c_int
def Utctai(utc1, utc2):
    """ Timescale transformation: Coordinated Universal Time (UTC) to
    International Atomic Time (TAI).

    :param utc1, utc2: UTC as a two-part Julian Date.
    :type utc1, utc2: float

    :returns: a tuple of three items:

        * TAI as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 243*
    """
    tai1 = c_double()
    tai2 = c_double()
    s = sofa.iauUtctai(utc1, utc2, byref(tai1), byref(tai2))
    return tai1.value, tai2.value, s


# iauUtcut1
sofa.iauUtcut1.argtypes = [c_double, #utc1
                            c_double, #utc2
                            c_double, #dut1
                            POINTER(c_double), #ut11
                            POINTER(c_double)] #ut12
sofa.iauUtcut1.restype = c_int
def Utcut1(utc1, utc2, dut1):
    """ Timescale transformation: Coordinated Universal Time (UTC) to
    Universal Time (UT1)

    :param utc1, utc2: UTC as a two-part Julian Date.
    :type utc1, utc2: float

    :param dut1: UT1-UTC in seconds.
    :type dut1: float

    :returns: a tuple of three items:

        * UT1 as a two-part Julian Date
        * function status.

    *sofa manual.pdf page 244*
    """
    ut11 = c_double()
    ut12 = c_double()
    s = sofa.iauUtcut1(utc1, utc2, dut1, byref(ut11), byref(ut12))
    return ut11.value, ut12.value, s


# iauXy06
sofa.iauXy06.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #x
                            POINTER(c_double)] #y
def Xy06(date1, date2):
    """ X,Y coordinates of the celestial intermediate pole from series 
    based on IAU 2006 precession and IAU 2000A nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple containing X and Y CIP coordinates.

    *sofa manual.pdf page 246*
    """
    x = c_double()
    y = c_double()
    sofa.iauXy06(date1, date2, byref(x), byref(y))
    return x.value, y.value


# iauXys00a
sofa.iauXys00a.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #x
                            POINTER(c_double), #y
                            POINTER(c_double)] #s
def Xys00a(date1, date2):
    """ For a given TT date, compute X, Y coordinates of the celestial 
    intermediate pole and the CIO locator *s*, using IAU 2000A 
    precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 3-tuple:

        * X CIP coordinate
        * Y CIP coordinate
        * the CIO locator *s*.

    *sofa manual.pdf page 248*
    """
    x = c_double()
    y = c_double()
    s = c_double()
    sofa.iauXys00a(date1, date2, byref(x), byref(y), byref(s))
    return x.value, y.value, s.value


# iauXys00b
sofa.iauXys00b.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #x
                            POINTER(c_double), #y
                            POINTER(c_double)] #s
def Xys00b(date1, date2):
    """ For a given TT date, compute X, Y coordinates of the celestial 
    intermediate pole and the CIO locator *s*, using IAU 2000B 
    precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 3-tuple:

        * X CIP coordinate
        * Y CIP coordinate
        * the CIO locator *s*.

    *sofa manual.pdf page 249*
    """
    x = c_double()
    y = c_double()
    s = c_double()
    sofa.iauXys00b(date1, date2, byref(x), byref(y), byref(s))
    return x.value, y.value, s.value


# iauXys06a
sofa.iauXys06a.argtypes = [c_double, #date1
                            c_double, #date2
                            POINTER(c_double), #x
                            POINTER(c_double), #y
                            POINTER(c_double)] #s
def Xys06a(date1, date2):
    """ For a given TT date, compute X, Y coordinates of the celestial 
    intermediate pole and the CIO locator *s*, using IAU 2006 precession 
    and IAU 2000A nutation models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 3-tuple:

        * X CIP coordinate
        * Y CIP coordinate
        * the CIO locator *s*.

    *sofa manual.pdf page 250*
    """
    x = c_double()
    y = c_double()
    s = c_double()
    sofa.iauXys06a(date1, date2, byref(x), byref(y), byref(s))
    return x.value, y.value, s.value


# iauZp
sofa.iauZp.argtypes = [ndpointer(shape=(1,3), dtype=float)] #p
def Zp(p):
    """ Zero a p-vector.

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: p-vector filled with zeros. Note that if *p* is supplied as a \
        numpy.ndarray it will be modified in-place. Otherwise a new array is \
        created and returned.

    *sofa manual.pdf page 251*
    """
    p2 = asmatrix(p, dtype=float)
    sofa.iauZp(p2)
    return p2


# iauZpv
sofa.iauZpv.argtypes = [ndpointer(shape=(2,3), dtype=float)] #pv
def Zpv(pv):
    """ Zero a pv-vector.

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: pv-vector filled with zeros. Note that if *pv* is supplied as a\
        numpy.ndarray it will be modified in-place. Otherwise a new array is \
        created and returned.

    *sofa manual.pdf page 252*
    """
    pv2 = asmatrix(pv, dtype=float)
    sofa.iauZpv(pv2)
    return pv2


# iauZr
sofa.iauZr.argtypes = [ndpointer(shape=(3,3), dtype=float)] #r
def Zr(r):
    """ Initialize a rotation matrix to the null matrix.

    :param r: rotation matrix.
    :type r: array-like shape (3,3)

    :returns: rotation matrix. Note that if *r* is supplied as a\
        numpy.ndarray it will be modified in-place. Otherwise a new array is \
        created and returned.

    *sofa manual.pdf page 253*
    """
    r2 = asmatrix(r, dtype=float)
    sofa.iauZr(r2)
    return r2

