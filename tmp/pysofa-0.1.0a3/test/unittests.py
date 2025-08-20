# -*- coding: utf-8 -*-

#
# Copyright 2010 Frédéric Grollier
#
# Distributed under the terms of the MIT license
#

from __future__ import division

import unittest
import sys
import pysofa
from numpy import ndarray
from numpy import array


def assertAlmostEqual(inst, first, second, places=7, msg=u'', delta=None):
    if delta is None:
        return inst.assertAlmostEqualLegacy(first, second, places=places,
                                                msg=msg)

    a = first - second
    if abs(a) > delta:
        f = abs(second / a)
        raise inst.failureException, \
            (msg or "wanted %r got %r (1/%r)" % (second, first, f))

if sys.version_info < (2, 7, 0):
    unittest.TestCase.assertAlmostEqualLegacy = \
                                            unittest.TestCase.assertAlmostEqual
    unittest.TestCase.assertAlmostEqual = assertAlmostEqual



class TestAstronomy(unittest.TestCase):
    def test_A2af(self):
        sign, idmsf = pysofa.A2af(4, 2.345)

        self.assertEqual(sign, '+')

        self.assertEqual(idmsf[0],  134)
        self.assertEqual(idmsf[1],   21)
        self.assertEqual(idmsf[2],   30)
        self.assertEqual(idmsf[3], 9706)

    def test_A2tf(self):
        sign, ihmsf = pysofa.A2tf(4, -3.01234)

        self.assertEqual(sign, '-')

        self.assertEqual(ihmsf[0],   11)
        self.assertEqual(ihmsf[1],   30)
        self.assertEqual(ihmsf[2],   22)
        self.assertEqual(ihmsf[3], 6484,)

    def test_Af2a(self):
        rad, s = pysofa.Af2a('-', 45, 13, 27.2)
        self.assertAlmostEqual(rad, -0.7893115794313644842, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Anp(self):
        r = pysofa.Anp(-0.1)
        self.assertAlmostEqual(r, 6.183185307179586477, delta=1e-12)

    def test_Anpm(self):
        r = pysofa.Anpm(-4.0)
        self.assertAlmostEqual(r, 2.283185307179586477, delta=1e-12)

    def test_Bi00(self):
        """ Ok.
        """
        dpsibi, depsbi, dra = pysofa.Bi00()
        self.assertAlmostEqual(dpsibi, -0.2025309152835086613e-6, delta=1e-12)
        self.assertAlmostEqual(depsbi, -0.3306041454222147847e-7, delta=1e-12)
        self.assertAlmostEqual(dra, -0.7078279744199225506e-7, delta=1e-12)

    def test_Bp00(self):
        rb, rp, rbp = pysofa.Bp00(2400000.5, 50123.9999)
        self.assertAlmostEqual(rb[0,0], 0.9999999999999942498, delta=1e-12)
        self.assertAlmostEqual(rb[0,1], -0.7078279744199196626e-7, delta=1e-16)
        self.assertAlmostEqual(rb[0,2], 0.8056217146976134152e-7, delta=1e-16)
        self.assertAlmostEqual(rb[1,0], 0.7078279477857337206e-7, delta=1e-16)
        self.assertAlmostEqual(rb[1,1], 0.9999999999999969484, delta=1e-12)
        self.assertAlmostEqual(rb[1,2], 0.3306041454222136517e-7, delta=1e-16)
        self.assertAlmostEqual(rb[2,0], -0.8056217380986972157e-7, delta=1e-16)
        self.assertAlmostEqual(rb[2,1], -0.3306040883980552500e-7, delta=1e-16)
        self.assertAlmostEqual(rb[2,2], 0.9999999999999962084, delta=1e-12)

        self.assertAlmostEqual(rp[0,0], 0.9999995504864048241, delta=1e-12)
        self.assertAlmostEqual(rp[0,1], 0.8696113836207084411e-3, delta=1e-14)
        self.assertAlmostEqual(rp[0,2], 0.3778928813389333402e-3, delta=1e-14)
        self.assertAlmostEqual(rp[1,0], -0.8696113818227265968e-3, delta=1e-14)
        self.assertAlmostEqual(rp[1,1], 0.9999996218879365258, delta=1e-12)
        self.assertAlmostEqual(rp[1,2], -0.1690679263009242066e-6, delta=1e-14)
        self.assertAlmostEqual(rp[2,0], -0.3778928854764695214e-3, delta=1e-14)
        self.assertAlmostEqual(rp[2,1], -0.1595521004195286491e-6, delta=1e-14)
        self.assertAlmostEqual(rp[2,2], 0.9999999285984682756, delta=1e-12)

        self.assertAlmostEqual(rbp[0,0], 0.9999995505175087260, delta=1e-12)
        self.assertAlmostEqual(rbp[0,1], 0.8695405883617884705e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[0,2], 0.3779734722239007105e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[1,0], -0.8695405990410863719e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[1,1], 0.9999996219494925900, delta=1e-12)
        self.assertAlmostEqual(rbp[1,2], -0.1360775820404982209e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,0], -0.3779734476558184991e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[2,1], -0.1925857585832024058e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,2], 0.9999999285680153377, delta=1e-12)

    def test_Bp06(self):
        rb, rp, rbp = pysofa.Bp06(2400000.5, 50123.9999)
        self.assertAlmostEqual(rb[0,0], 0.9999999999999942497, delta=1e-12)
        self.assertAlmostEqual(rb[0,1], -0.7078368960971557145e-7, delta=1e-14)
        self.assertAlmostEqual(rb[0,2], 0.8056213977613185606e-7, delta=1e-14)
        self.assertAlmostEqual(rb[1,0], 0.7078368694637674333e-7, delta=1e-14)
        self.assertAlmostEqual(rb[1,1], 0.9999999999999969484, delta=1e-12)
        self.assertAlmostEqual(rb[1,2], 0.3305943742989134124e-7, delta=1e-14)
        self.assertAlmostEqual(rb[2,0], -0.8056214211620056792e-7, delta=1e-14)
        self.assertAlmostEqual(rb[2,1], -0.3305943172740586950e-7, delta=1e-14)
        self.assertAlmostEqual(rb[2,2], 0.9999999999999962084, delta=1e-12)

        self.assertAlmostEqual(rp[0,0], 0.9999995504864960278, delta=1e-12)
        self.assertAlmostEqual(rp[0,1], 0.8696112578855404832e-3, delta=1e-14)
        self.assertAlmostEqual(rp[0,2], 0.3778929293341390127e-3, delta=1e-14)
        self.assertAlmostEqual(rp[1,0], -0.8696112560510186244e-3, delta=1e-14)
        self.assertAlmostEqual(rp[1,1], 0.9999996218880458820, delta=1e-12)
        self.assertAlmostEqual(rp[1,2], -0.1691646168941896285e-6, delta=1e-14)
        self.assertAlmostEqual(rp[2,0], -0.3778929335557603418e-3, delta=1e-14)
        self.assertAlmostEqual(rp[2,1], -0.1594554040786495076e-6, delta=1e-14)
        self.assertAlmostEqual(rp[2,2], 0.9999999285984501222, delta=1e-12)

        self.assertAlmostEqual(rbp[0,0], 0.9999995505176007047, delta=1e-12)
        self.assertAlmostEqual(rbp[0,1], 0.8695404617348208406e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[0,2], 0.3779735201865589104e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[1,0], -0.8695404723772031414e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[1,1], 0.9999996219496027161, delta=1e-12)
        self.assertAlmostEqual(rbp[1,2], -0.1361752497080270143e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,0], -0.3779734957034089490e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[2,1], -0.1924880847894457113e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,2], 0.9999999285679971958, delta=1e-12)

    def test_Bpn2xy(self):
        #rbpn = ndarray(shape=(3,3))
        #rbpn[0][0] =  9.999962358680738e-1
        #rbpn[0][1] = 9.999962358680738e-1
        #rbpn[0][2] = -1.093569785342370e-3

        #rbpn[1][0] =  2.516462370370876e-3
        #rbpn[1][1] =  2.516462370370876e-3
        #rbpn[1][2] =  4.006159587358310e-5

        #rbpn[2][0] =  1.093465510215479e-3
        #rbpn[2][1] = -4.281337229063151e-5
        #rbpn[2][2] =  9.999994012499173e-1
        rbpn = ((9.999962358680738e-1, 9.999962358680738e-1, -1.093569785342370e-3),
                (2.516462370370876e-3, 2.516462370370876e-3, 4.006159587358310e-5),
                (1.093465510215479e-3, -4.281337229063151e-5, 9.999994012499173e-1))

        x, y = pysofa.Bpn2xy(rbpn)
        self.assertAlmostEqual(x,  1.093465510215479e-3, delta=1e-12)
        self.assertAlmostEqual(y, -4.281337229063151e-5, delta=1e-12)

    def test_C2i00a(self):
        rc2i = pysofa.C2i00a(2400000.5, 53736.0)

        self.assertAlmostEqual(rc2i[0,0], 0.9999998323037165557, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,1], 0.5581526348992140183e-9, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,2], -0.5791308477073443415e-3, delta=1e-12)

        self.assertAlmostEqual(rc2i[1,0], -0.2384266227870752452e-7, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,1], 0.9999999991917405258, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,2], -0.4020594955028209745e-4, delta=1e-12)

        self.assertAlmostEqual(rc2i[2,0], 0.5791308472168152904e-3, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,1], 0.4020595661591500259e-4, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,2], 0.9999998314954572304, delta=1e-12)

    def test_C2i00b(self):
        rc2i = pysofa.C2i00b(2400000.5, 53736.0)

        self.assertAlmostEqual(rc2i[0,0], 0.9999998323040954356, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,1], 0.5581526349131823372e-9, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,2], -0.5791301934855394005e-3, delta=1e-12)

        self.assertAlmostEqual(rc2i[1,0], -0.2384239285499175543e-7, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,1], 0.9999999991917574043, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,2], -0.4020552974819030066e-4, delta=1e-12)

        self.assertAlmostEqual(rc2i[2,0], 0.5791301929950208873e-3, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,1], 0.4020553681373720832e-4, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,2], 0.9999998314958529887, delta=1e-12)

    def test_C2i06a(self):
        rc2i = pysofa.C2i06a(2400000.5, 53736.0)

        self.assertAlmostEqual(rc2i[0,0], 0.9999998323037159379, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,1], 0.5581121329587613787e-9, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,2], -0.5791308487740529749e-3, delta=1e-12)

        self.assertAlmostEqual(rc2i[1,0], -0.2384253169452306587e-7, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,1], 0.9999999991917467827, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,2], -0.4020579392895682558e-4, delta=1e-12)

        self.assertAlmostEqual(rc2i[2,0], 0.5791308482835292617e-3, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,1], 0.4020580099454020310e-4, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,2], 0.9999998314954628695, delta=1e-12)

    def test_C2ibpn(self):
        rbpn = ndarray(shape=(3,3))
        rbpn[0][0] =  9.999962358680738e-1
        rbpn[0][1] = -2.516417057665452e-3
        rbpn[0][2] = -1.093569785342370e-3

        rbpn[1][0] =  2.516462370370876e-3
        rbpn[1][1] =  9.999968329010883e-1
        rbpn[1][2] =  4.006159587358310e-5

        rbpn[2][0] =  1.093465510215479e-3
        rbpn[2][1] = -4.281337229063151e-5
        rbpn[2][2] =  9.999994012499173e-1

        rc2i = pysofa.C2ibpn(2400000.5, 50123.9999, rbpn)

        self.assertAlmostEqual(rc2i[0,0], 0.9999994021664089977, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,1], -0.3869195948017503664e-8, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,2], -0.1093465511383285076e-2, delta=1e-12)

        self.assertAlmostEqual(rc2i[1,0], 0.5068413965715446111e-7, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,1], 0.9999999990835075686, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,2], 0.4281334246452708915e-4, delta=1e-12)

        self.assertAlmostEqual(rc2i[2,0], 0.1093465510215479000e-2, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,1], -0.4281337229063151000e-4, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,2], 0.9999994012499173103, delta=1e-12)

    def test_C2ixy(self):
        x = 0.5791308486706011000e-3
        y = 0.4020579816732961219e-4

        rc2i = pysofa.C2ixy(2400000.5, 53736, x, y)

        self.assertAlmostEqual(rc2i[0,0], 0.9999998323037157138, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,1], 0.5581526349032241205e-9, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,2], -0.5791308491611263745e-3, delta=1e-12)

        self.assertAlmostEqual(rc2i[1,0], -0.2384257057469842953e-7, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,1], 0.9999999991917468964, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,2], -0.4020579110172324363e-4, delta=1e-12)

        self.assertAlmostEqual(rc2i[2,0], 0.5791308486706011000e-3, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,1], 0.4020579816732961219e-4, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,2], 0.9999998314954627590, delta=1e-12)

    def test_C2ixys(self):
        x =  0.5791308486706011000e-3
        y =  0.4020579816732961219e-4
        s = -0.1220040848472271978e-7

        rc2i = pysofa.C2ixys(x, y, s)

        self.assertAlmostEqual(rc2i[0,0], 0.9999998323037157138, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,1], 0.5581984869168499149e-9, delta=1e-12)
        self.assertAlmostEqual(rc2i[0,2], -0.5791308491611282180e-3, delta=1e-12)

        self.assertAlmostEqual(rc2i[1,0], -0.2384261642670440317e-7, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,1], 0.9999999991917468964, delta=1e-12)
        self.assertAlmostEqual(rc2i[1,2], -0.4020579110169668931e-4, delta=1e-12)

        self.assertAlmostEqual(rc2i[2,0], 0.5791308486706011000e-3, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,1], 0.4020579816732961219e-4, delta=1e-12)
        self.assertAlmostEqual(rc2i[2,2], 0.9999998314954627590, delta=1e-12)

    def test_C2s(self):
        p = array([100.0, -50.0, 25.0])

        theta, phi = pysofa.C2s(p)

        self.assertAlmostEqual(theta, -0.4636476090008061162, delta=1e-14)
        self.assertAlmostEqual(phi, 0.2199879773954594463, delta=1e-14)

    def test_C2t00a(self):
        tta = 2400000.5
        uta = 2400000.5
        ttb = 53736.0
        utb = 53736.0
        xp = 2.55060238e-7
        yp = 1.860359247e-6

        rc2t = pysofa.C2t00a(tta, ttb, uta, utb, xp, yp)

        self.assertAlmostEqual(rc2t[0,0], -0.1810332128307182668, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,1], 0.9834769806938457836, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,2], 0.6555535638688341725e-4, delta=1e-12)

        self.assertAlmostEqual(rc2t[1,0], -0.9834768134135984552, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,1], -0.1810332203649520727, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,2], 0.5749801116141056317e-3, delta=1e-12)

        self.assertAlmostEqual(rc2t[2,0], 0.5773474014081406921e-3, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,1], 0.3961832391770163647e-4, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,2], 0.9999998325501692289, delta=1e-12)

    def test_C2t00b(self):
        tta = 2400000.5
        uta = 2400000.5
        ttb = 53736.0
        utb = 53736.0
        xp = 2.55060238e-7
        yp = 1.860359247e-6

        rc2t = pysofa.C2t00b(tta, ttb, uta, utb, xp, yp)

        self.assertAlmostEqual(rc2t[0,0], -0.1810332128439678965, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,1], 0.9834769806913872359, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,2], 0.6555565082458415611e-4, delta=1e-12)

        self.assertAlmostEqual(rc2t[1,0], -0.9834768134115435923, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,1], -0.1810332203784001946, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,2], 0.5749793922030017230e-3, delta=1e-12)

        self.assertAlmostEqual(rc2t[2,0], 0.5773467471863534901e-3, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,1], 0.3961790411549945020e-4, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,2], 0.9999998325505635738, delta=1e-12)

    def test_C2t06a(self):
        tta = 2400000.5
        uta = 2400000.5
        ttb = 53736.0
        utb = 53736.0
        xp = 2.55060238e-7
        yp = 1.860359247e-6

        rc2t = pysofa.C2t06a(tta, ttb, uta, utb, xp, yp)

        self.assertAlmostEqual(rc2t[0,0], -0.1810332128305897282, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,1], 0.9834769806938592296, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,2], 0.6555550962998436505e-4, delta=1e-12)

        self.assertAlmostEqual(rc2t[1,0], -0.9834768134136214897, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,1], -0.1810332203649130832, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,2], 0.5749800844905594110e-3, delta=1e-12)

        self.assertAlmostEqual(rc2t[2,0], 0.5773474024748545878e-3, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,1], 0.3961816829632690581e-4, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,2], 0.9999998325501747785, delta=1e-12)

    def test_C2tcio(self):
        rc2i = ndarray(shape=(3,3))
        rc2i[0][0] =  0.9999998323037164738
        rc2i[0][1] =  0.5581526271714303683e-9
        rc2i[0][2] = -0.5791308477073443903e-3

        rc2i[1][0] = -0.2384266227524722273e-7
        rc2i[1][1] =  0.9999999991917404296
        rc2i[1][2] = -0.4020594955030704125e-4

        rc2i[2][0] =  0.5791308472168153320e-3
        rc2i[2][1] =  0.4020595661593994396e-4
        rc2i[2][2] =  0.9999998314954572365

        era = 1.75283325530307

        rpom = ndarray(shape=(3,3))
        rpom[0][0] =  0.9999999999999674705
        rpom[0][1] = -0.1367174580728847031e-10
        rpom[0][2] =  0.2550602379999972723e-6

        rpom[1][0] =  0.1414624947957029721e-10
        rpom[1][1] =  0.9999999999982694954
        rpom[1][2] = -0.1860359246998866338e-5

        rpom[2][0] = -0.2550602379741215275e-6
        rpom[2][1] =  0.1860359247002413923e-5
        rpom[2][2] =  0.9999999999982369658

        rc2t = pysofa.C2tcio(rc2i, era, rpom)

        self.assertAlmostEqual(rc2t[0,0], -0.1810332128307110439, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,1], 0.9834769806938470149, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,2], 0.6555535638685466874e-4, delta=1e-12)

        self.assertAlmostEqual(rc2t[1,0], -0.9834768134135996657, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,1], -0.1810332203649448367, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,2], 0.5749801116141106528e-3, delta=1e-12)

        self.assertAlmostEqual(rc2t[2,0], 0.5773474014081407076e-3, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,1], 0.3961832391772658944e-4, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,2], 0.9999998325501691969, delta=1e-12)

    def test_C2teqx(self):
        rbpn = ndarray(shape=(3,3))
        rbpn[0][0] =  0.9999989440476103608
        rbpn[0][1] = -0.1332881761240011518e-2
        rbpn[0][2] = -0.5790767434730085097e-3

        rbpn[1][0] =  0.1332858254308954453e-2
        rbpn[1][1] =  0.9999991109044505944
        rbpn[1][2] = -0.4097782710401555759e-4

        rbpn[2][0] =  0.5791308472168153320e-3
        rbpn[2][1] =  0.4020595661593994396e-4
        rbpn[2][2] =  0.9999998314954572365

        gst = 1.754166138040730516

        rpom = ndarray(shape=(3,3))
        rpom[0][0] =  0.9999999999999674705
        rpom[0][1] = -0.1367174580728847031e-10
        rpom[0][2] =  0.2550602379999972723e-6

        rpom[1][0] =  0.1414624947957029721e-10
        rpom[1][1] =  0.9999999999982694954
        rpom[1][2] = -0.1860359246998866338e-5

        rpom[2][0] = -0.2550602379741215275e-6
        rpom[2][1] =  0.1860359247002413923e-5
        rpom[2][2] =  0.9999999999982369658

        rc2t = pysofa.C2teqx(rbpn, gst, rpom)

        self.assertAlmostEqual(rc2t[0,0], -0.1810332128528685730, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,1], 0.9834769806897685071, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,2], 0.6555535639982634449e-4, delta=1e-12)

        self.assertAlmostEqual(rc2t[1,0], -0.9834768134095211257, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,1], -0.1810332203871023800, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,2], 0.5749801116126438962e-3, delta=1e-12)

        self.assertAlmostEqual(rc2t[2,0], 0.5773474014081539467e-3, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,1], 0.3961832391768640871e-4, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,2], 0.9999998325501691969, delta=1e-12)

    def test_C2tpe(self):
        tta = 2400000.5
        uta = 2400000.5
        ttb = 53736.0
        utb = 53736.0
        deps =  0.4090789763356509900
        dpsi = -0.9630909107115582393e-5
        xp = 2.55060238e-7
        yp = 1.860359247e-6

        rc2t = pysofa.C2tpe(tta, ttb, uta, utb, dpsi, deps, xp, yp)

        self.assertAlmostEqual(rc2t[0,0], -0.1813677995763029394, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,1], 0.9023482206891683275, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,2], -0.3909902938641085751, delta=1e-12)

        self.assertAlmostEqual(rc2t[1,0], -0.9834147641476804807, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,1], -0.1659883635434995121, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,2], 0.7309763898042819705e-1, delta=1e-12)

        self.assertAlmostEqual(rc2t[2,0], 0.1059685430673215247e-2, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,1], 0.3977631855605078674, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,2], 0.9174875068792735362, delta=1e-12)

    def test_C2txy(self):
        tta = 2400000.5
        uta = 2400000.5
        ttb = 53736.0
        utb = 53736.0
        x = 0.5791308486706011000e-3
        y = 0.4020579816732961219e-4
        xp = 2.55060238e-7
        yp = 1.860359247e-6

        rc2t = pysofa.C2txy(tta, ttb, uta, utb, x, y, xp, yp)

        self.assertAlmostEqual(rc2t[0,0], -0.1810332128306279253, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,1], 0.9834769806938520084, delta=1e-12)
        self.assertAlmostEqual(rc2t[0,2], 0.6555551248057665829e-4, delta=1e-12)

        self.assertAlmostEqual(rc2t[1,0], -0.9834768134136142314, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,1], -0.1810332203649529312, delta=1e-12)
        self.assertAlmostEqual(rc2t[1,2], 0.5749800843594139912e-3, delta=1e-12)

        self.assertAlmostEqual(rc2t[2,0], 0.5773474028619264494e-3, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,1], 0.3961816546911624260e-4, delta=1e-12)
        self.assertAlmostEqual(rc2t[2,2], 0.9999998325501746670, delta=1e-12)

    def test_Cal2jd(self):
        djm0, djm, status = pysofa.Cal2jd(2003, 06, 01)

        self.assertAlmostEqual(djm0, 2400000.5, delta=0.0)
        self.assertAlmostEqual(djm, 52791.0, delta=0.0)

        self.assertEqual(status, 0)

    def test_Cp(self):
        p = array((0.3, 1.2, -2.5))

        c = pysofa.Cp(p)

        self.assertAlmostEqual(c[0,0], 0.3, delta=0.0)
        self.assertAlmostEqual(c[0,1], 1.2, delta=0.0)
        self.assertAlmostEqual(c[0,2], -2.5, delta=0.0)

    def test_Cpv(self):
        p = array(((0.3, 1.2, -2.5), (-0.5, 3.1, 0.9)))

        c = pysofa.Cpv(p)

        self.assertTrue((p == c).all())

    def test_Cr(self):
        p = array(((2.0, 3.0, 2.0), (3.0, 2.0, 3.0), (3.0, 4., 5.0)))

        c = pysofa.Cr(p)

        self.assertTrue((p == c).all())

    def test_D2dtf(self):
        iy, im, id, h, m, s, f, j = pysofa.D2dtf("UTC", 5, 2400000.5, 
                                                    49533.99999)
        self.assertEqual(iy, 1994)
        self.assertEqual(im, 6)
        self.assertEqual(id, 30)
        self.assertEqual(h, 23)
        self.assertEqual(m, 59)
        self.assertEqual(s, 60)
        self.assertEqual(f, 13599)
        self.assertEqual(j, 0)

    def test_D2tf(self):
        sign, ihmsf = pysofa.D2tf(4, -0.987654321)

        self.assertEqual(sign, '-')

        self.assertEqual(ihmsf[0], 23)
        self.assertEqual(ihmsf[1], 42)
        self.assertEqual(ihmsf[2], 13)
        self.assertEqual(ihmsf[3], 3333)

    def test_Dat(self):
        deltat, status = pysofa.Dat(2003, 6, 1, 0)

        self.assertAlmostEqual(deltat, 32.0, delta=0.0)
        self.assertEqual(status, 0)

        deltat, status = pysofa.Dat(2008, 1, 17, 0)

        self.assertAlmostEqual(deltat, 33.0, delta=0.0)
        self.assertEqual(status, 0)

    def test_Dtdb(self):
        dtdb = pysofa.Dtdb(2448939.5, 0.123, 0.76543, 5.0123, 5525.242, 3190.0)

        self.assertAlmostEqual(dtdb, -0.1280368005936998991e-2, delta=1e-15)

    def test_Dtf2d(self):
        u1, u2, s = pysofa.Dtf2d("UTC", 1994, 6, 30, 23, 59, 60.13599)

        self.assertAlmostEqual(u1+u2, 2449534.49999, delta=1e-6)
        self.assertEqual(s, 0)

    def test_Ee00(self):
        epsa =  0.4090789763356509900
        dpsi = -0.9630909107115582393e-5

        ee = pysofa.Ee00(2400000.5, 53736.0, epsa, dpsi)

        self.assertAlmostEqual(ee, -0.8834193235367965479e-5, delta=1e-18)

    def test_Ee00a(self):
        ee = pysofa.Ee00a(2400000.5, 53736.0)

        self.assertAlmostEqual(ee, -0.8834192459222588227e-5, delta=1e-18)

    def test_Ee00b(self):
        ee = pysofa.Ee00b(2400000.5, 53736.0)

        self.assertAlmostEqual(ee, -0.8835700060003032831e-5, delta=1e-18)

    def test_Ee06a(self):
        ee = pysofa.Ee06a(2400000.5, 53736.0)

        self.assertAlmostEqual(ee, -0.8834195072043790156e-5, delta=1e-15)

    def test_Eect00(self):
        eect = pysofa.Eect00(2400000.5, 53736.0)

        self.assertAlmostEqual(eect, 0.2046085004885125264e-8, delta=1e-20)

    def test_Eform(self):

        a, f, j = pysofa.Eform(0)

        self.assertEqual(j, -1)

        a, f, j = pysofa.Eform(1)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(a, 6378137.0, delta=1e-10)
        self.assertAlmostEqual(f, 0.0033528106647474807, delta=1e-18)

        a, f, j = pysofa.Eform(2)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(a, 6378137.0, delta=1e-10)
        self.assertAlmostEqual(f, 0.0033528106811823189, delta=1e-18)

        a, f, j = pysofa.Eform(3)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(a, 6378135.0, delta=1e-10)
        self.assertAlmostEqual(f, 0.0033527794541675049, delta=1e-18)

        a, f, j = pysofa.Eform(4)
        self.assertEqual(j, -1)

    def test_Eo06a(self):
        eo = pysofa.Eo06a(2400000.5, 53736.0)

        self.assertAlmostEqual(eo, -0.1332882371941833644e-2, delta=1e-15)

    def test_Eors(self):
        rnpb = ndarray(shape=(3,3))


        rnpb[0][0] =  0.9999989440476103608
        rnpb[0][1] = -0.1332881761240011518e-2
        rnpb[0][2] = -0.5790767434730085097e-3

        rnpb[1][0] =  0.1332858254308954453e-2
        rnpb[1][1] =  0.9999991109044505944
        rnpb[1][2] = -0.4097782710401555759e-4

        rnpb[2][0] =  0.5791308472168153320e-3
        rnpb[2][1] =  0.4020595661593994396e-4
        rnpb[2][2] =  0.9999998314954572365

        s = -0.1220040848472271978e-7

        eo = pysofa.Eors(rnpb, s)

        self.assertAlmostEqual(eo, -0.1332882715130744606e-2, delta=1e-14)

    def test_Epb(self):
        epb = pysofa.Epb(2415019.8135, 30103.18648)

        self.assertAlmostEqual(epb, 1982.418424159278580, delta=1e-12)

    def test_Epb2jd(self):
        epb = 1957.3

        djm0, djm = pysofa.Epb2jd(epb)

        self.assertAlmostEqual(djm0, 2400000.5, delta=1e-9)
        self.assertAlmostEqual(djm, 35948.1915101513, delta=1e-9)

    def test_Epj(self):
        epj = pysofa.Epj(2451545, -7392.5)

        self.assertAlmostEqual(epj, 1979.760438056125941, delta=1e-12)

    def test_Epj2jd(self):
        epj = 1996.8

        djm0, djm = pysofa.Epj2jd(epj)

        self.assertAlmostEqual(djm0, 2400000.5, delta=1e-9)
        self.assertAlmostEqual(djm, 50375.7, delta=1e-9)

    def test_Epv00(self):
        pvh, pvb, j = pysofa.Epv00(2400000.5, 53411.52501161)

        self.assertAlmostEqual(pvh[0,0], -0.7757238809297706813, delta=1e-14)
        self.assertAlmostEqual(pvh[0,1], 0.5598052241363340596, delta=1e-14)
        self.assertAlmostEqual(pvh[0,2], 0.2426998466481686993, delta=1e-14)

        self.assertAlmostEqual(pvh[1,0], -0.1091891824147313846e-1, delta=1e-15)
        self.assertAlmostEqual(pvh[1,1], -0.1247187268440845008e-1, delta=1e-15)
        self.assertAlmostEqual(pvh[1,2], -0.5407569418065039061e-2, delta=1e-15)

        self.assertAlmostEqual(pvb[0,0], -0.7714104440491111971, delta=1e-14)
        self.assertAlmostEqual(pvb[0,1], 0.5598412061824171323, delta=1e-14)
        self.assertAlmostEqual(pvb[0,2], 0.2425996277722452400, delta=1e-14)

        self.assertAlmostEqual(pvb[1,0], -0.1091874268116823295e-1, delta=1e-15)
        self.assertAlmostEqual(pvb[1,1], -0.1246525461732861538e-1, delta=1e-15)
        self.assertAlmostEqual(pvb[1,2], -0.5404773180966231279e-2, delta=1e-15)

        self.assertEqual(j, 0)

    def test_Eqeq94(self):
        eqeq = pysofa.Eqeq94(2400000.5, 41234.0)

        self.assertAlmostEqual(eqeq, 0.5357758254609256894e-4, delta=1e-17)

    def test_Era00(self):
        era00 = pysofa.Era00(2400000.5, 54388.0);

        self.assertAlmostEqual(era00, 0.4022837240028158102, delta=1e-12)

    def test_Fad03(self):
        self.assertAlmostEqual(pysofa.Fad03(0.80), 1.946709205396925672, delta=1e-12)

    def test_Fae03(self):
        self.assertAlmostEqual(pysofa.Fae03(0.80), 1.744713738913081846, delta=1e-12)

    def test_Faf03(self):
        self.assertAlmostEqual(pysofa.Faf03(0.80), 0.2597711366745499518, delta=1e-12)

    def test_Faju03(self):
        self.assertAlmostEqual(pysofa.Faju03(0.80), 5.275711665202481138, delta=1e-12)

    def test_Fal03(self):
        self.assertAlmostEqual(pysofa.Fal03(0.80), 5.132369751108684150, delta=1e-12)

    def test_Falp03(self):
        self.assertAlmostEqual(pysofa.Falp03(0.80), 6.226797973505507345, delta=1e-12)

    def test_Fama03(self):
        self.assertAlmostEqual(pysofa.Fama03(0.80), 3.275506840277781492, delta=1e-12)

    def test_Fame03(self):
        self.assertAlmostEqual(pysofa.Fame03(0.80), 5.417338184297289661, delta=1e-12)

    def test_Fane03(self):
        self.assertAlmostEqual(pysofa.Fane03(0.80), 2.079343830860413523, delta=1e-12)

    def test_Faom03(self):
        self.assertAlmostEqual(pysofa.Faom03(0.80), -5.973618440951302183, delta=1e-12)

    def test_Fapa03(self):
        self.assertAlmostEqual(pysofa.Fapa03(0.80), 0.1950884762240000000e-1, delta=1e-12)

    def test_Fasa03(self):
        self.assertAlmostEqual(pysofa.Fasa03(0.80), 5.371574539440827046, delta=1e-12)

    def test_Faur03(self):
        self.assertAlmostEqual(pysofa.Faur03(0.80), 5.180636450180413523, delta=1e-12)

    def test_Fave03(self):
        self.assertAlmostEqual(pysofa.Fave03(0.80), 3.424900460533758000, delta=1e-12)

    def test_Fk52h(self):
        r5  =  1.76779433
        d5  = -0.2917517103
        dr5 = -1.91851572e-7
        dd5 = -5.8468475e-6
        px5 =  0.379210
        rv5 = -7.6

        rh, dh, drh, ddh, pxh, rvh = pysofa.Fk52h(r5, d5, dr5, dd5, px5, rv5)

        self.assertAlmostEqual(rh, 1.767794226299947632, delta=1e-14)
        self.assertAlmostEqual(dh,  -0.2917516070530391757, delta=1e-14)
        self.assertAlmostEqual(drh, -0.19618741256057224e-6, delta=1e-19)
        self.assertAlmostEqual(ddh, -0.58459905176693911e-5, delta=1e-19)
        self.assertAlmostEqual(pxh,  0.37921, delta=1e-14)
        self.assertAlmostEqual(rvh, -7.6000000940000254, delta=1e-11)

    def test_Fk5hip(self):
        r5h, s5h = pysofa.Fk5hip()

        self.assertAlmostEqual(r5h[0,0], 0.9999999999999928638, delta=1e-14)
        self.assertAlmostEqual(r5h[0,1], 0.1110223351022919694e-6, delta=1e-17)
        self.assertAlmostEqual(r5h[0,2], 0.4411803962536558154e-7, delta=1e-17)
        self.assertAlmostEqual(r5h[1,0], -0.1110223308458746430e-6, delta=1e-17)
        self.assertAlmostEqual(r5h[1,1], 0.9999999999999891830, delta=1e-14)
        self.assertAlmostEqual(r5h[1,2], -0.9647792498984142358e-7, delta=1e-17)
        self.assertAlmostEqual(r5h[2,0], -0.4411805033656962252e-7, delta=1e-17)
        self.assertAlmostEqual(r5h[2,1], 0.9647792009175314354e-7, delta=1e-17)
        self.assertAlmostEqual(r5h[2,2], 0.9999999999999943728, delta=1e-14)
        self.assertAlmostEqual(s5h[0,0], -0.1454441043328607981e-8, delta=1e-17)
        self.assertAlmostEqual(s5h[0,1], 0.2908882086657215962e-8, delta=1e-17)
        self.assertAlmostEqual(s5h[0,2], 0.3393695767766751955e-8, delta=1e-17)

    def test_Fk5hz(self):
        r5 =  1.76779433
        d5 = -0.2917517103

        rh, dh = pysofa.Fk5hz(r5, d5, 2400000.5, 54479.0)

        self.assertAlmostEqual(rh,  1.767794191464423978, delta=1e-12)
        self.assertAlmostEqual(dh, -0.2917516001679884419, delta=1e-12)

    def test_Fw2m(self):
        gamb = -0.2243387670997992368e-5
        phib =  0.4091014602391312982
        psi  = -0.9501954178013015092e-3
        eps  =  0.4091014316587367472

        r = pysofa.Fw2m(gamb, phib, psi, eps)

        self.assertAlmostEqual(r[0,0], 0.9999995505176007047, delta=1e-12)
        self.assertAlmostEqual(r[0,1], 0.8695404617348192957e-3, delta=1e-12)
        self.assertAlmostEqual(r[0,2], 0.3779735201865582571e-3, delta=1e-12)

        self.assertAlmostEqual(r[1,0], -0.8695404723772016038e-3, delta=1e-12)
        self.assertAlmostEqual(r[1,1], 0.9999996219496027161, delta=1e-12)
        self.assertAlmostEqual(r[1,2], -0.1361752496887100026e-6, delta=1e-12)

        self.assertAlmostEqual(r[2,0], -0.3779734957034082790e-3, delta=1e-12)
        self.assertAlmostEqual(r[2,1], -0.1924880848087615651e-6, delta=1e-12)
        self.assertAlmostEqual(r[2,2], 0.9999999285679971958, delta=1e-12)

    def test_Fw2xy(self):
        gamb = -0.2243387670997992368e-5
        phib =  0.4091014602391312982
        psi  = -0.9501954178013015092e-3
        eps  =  0.4091014316587367472

        x, y = pysofa.Fw2xy(gamb, phib, psi, eps)

        self.assertAlmostEqual(x, -0.3779734957034082790e-3, delta=1e-14)
        self.assertAlmostEqual(y, -0.1924880848087615651e-6, delta=1e-14)

    def test_Gc2gd(self):
        xyz = array((2e6, 3e6, 5.244e6))
        e, p, h, j = pysofa.Gc2gd( 0, xyz)

        self.assertEqual(j, -1)

        e, p, h, j = pysofa.Gc2gd( 1, xyz)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(e, 0.98279372324732907, delta=1e-14)
        self.assertAlmostEqual(p, 0.97160184819075459, delta=1e-14)
        self.assertAlmostEqual(h, 331.41724614260599, delta=1e-8)

        e, p, h, j = pysofa.Gc2gd( 2, xyz)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(e, 0.98279372324732907, delta=1e-14)
        self.assertAlmostEqual(p, 0.97160184820607853, delta=1e-14)
        self.assertAlmostEqual(h, 331.41731754844348, delta=1e-8)

        e, p, h, j = pysofa.Gc2gd( 3, xyz)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(e, 0.98279372324732907, delta=1e-14)
        self.assertAlmostEqual(p, 0.97160181811015119, delta=1e-14)
        self.assertAlmostEqual(h, 333.27707261303181, delta=1e-8)

        e, p, h, j = pysofa.Gc2gd( 4, xyz)

        self.assertEqual(j, -1)

    def test_Gc2gde(self):
        a = 6378136.0
        f = 0.0033528
        xyz = array((2e6, 3e6, 5.244e6))

        e, p, h, j = pysofa.Gc2gde(a, f, xyz)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(e, 0.98279372324732907, delta=1e-14)
        self.assertAlmostEqual(p, 0.97160183775704115, delta=1e-14)
        self.assertAlmostEqual(h, 332.36862495764397, delta=1e-8)

    def test_Gd2gc(self):
        e = 3.1
        p = -0.5
        h = 2500.0

        xyz, j = pysofa.Gd2gc( 0, e, p, h)

        self.assertEqual(j, -1)

        xyz, j = pysofa.Gd2gc( 1, e, p, h)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(xyz[0,0], -5599000.5577049947, delta=1e-7)
        self.assertAlmostEqual(xyz[0,1], 233011.67223479203, delta=1e-7)
        self.assertAlmostEqual(xyz[0,2], -3040909.4706983363, delta=1e-7)

        xyz, j = pysofa.Gd2gc( 2, e, p, h)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(xyz[0,0], -5599000.5577260984, delta=1e-7)
        self.assertAlmostEqual(xyz[0,1], 233011.6722356703, delta=1e-7)
        self.assertAlmostEqual(xyz[0,2], -3040909.4706095476, delta=1e-7)

        xyz, j = pysofa.Gd2gc( 3, e, p, h)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(xyz[0,0], -5598998.7626301490, delta=1e-7)
        self.assertAlmostEqual(xyz[0,1], 233011.5975297822, delta=1e-7)
        self.assertAlmostEqual(xyz[0,2], -3040908.6861467111, delta=1e-7)

        xyz, j = pysofa.Gd2gc( 4, e, p, h)

        self.assertEqual(j, -1)

    def test_Gd2gce(self):
        a = 6378136.0
        f = 0.0033528
        e = 3.1
        p = -0.5
        h = 2500.0

        xyz, j = pysofa.Gd2gce( a, f, e, p, h)

        self.assertEqual(j, 0)
        self.assertAlmostEqual(xyz[0,0], -5598999.6665116328, delta=1e-7)
        self.assertAlmostEqual(xyz[0,1], 233011.63514630572, delta=1e-7)
        self.assertAlmostEqual(xyz[0,2], -3040909.0517314132, delta=1e-7)

    def test_Gmst00(self):
        theta = pysofa.Gmst00(2400000.5, 53736.0, 2400000.5, 53736.0)

        self.assertAlmostEqual(theta, 1.754174972210740592, delta=1e-12)

    def test_Gmst06(self):
        theta = pysofa.Gmst06(2400000.5, 53736.0, 2400000.5, 53736.0)

        self.assertAlmostEqual(theta, 1.754174971870091203, delta=1e-12)

    def test_Gmst82(self):
        theta = pysofa.Gmst82(2400000.5, 53736.0)

        self.assertAlmostEqual(theta, 1.754174981860675096, delta=1e-12)

    def test_Gst00a(self):
        theta = pysofa.Gst00a(2400000.5, 53736.0, 2400000.5, 53736.0)

        self.assertAlmostEqual(theta, 1.754166138018281369, delta=1e-12)

    def test_Gst00b(self):
        theta = pysofa.Gst00b(2400000.5, 53736.0)

        self.assertAlmostEqual(theta, 1.754166136510680589, delta=1e-12)

    def test_Gst06(self):
        rnpb = ndarray(shape=(3,3))
        rnpb[0][0] =  0.9999989440476103608
        rnpb[0][1] = -0.1332881761240011518e-2
        rnpb[0][2] = -0.5790767434730085097e-3

        rnpb[1][0] =  0.1332858254308954453e-2
        rnpb[1][1] =  0.9999991109044505944
        rnpb[1][2] = -0.4097782710401555759e-4

        rnpb[2][0] =  0.5791308472168153320e-3
        rnpb[2][1] =  0.4020595661593994396e-4
        rnpb[2][2] =  0.9999998314954572365

        theta = pysofa.Gst06(2400000.5, 53736.0, 2400000.5, 53736.0, rnpb);

        self.assertAlmostEqual(theta, 1.754166138018167568, delta=1e-12)

    def test_Gst06a(self):
        theta = pysofa.Gst06a(2400000.5, 53736.0, 2400000.5, 53736.0)

        self.assertAlmostEqual(theta, 1.754166137675019159, delta=1e-12)

    def test_Gst94(self):
        theta = pysofa.Gst94(2400000.5, 53736.0)

        self.assertAlmostEqual(theta, 1.754166136020645203, delta=1e-12)

    def test_H2fk5(self):
        rh  =  1.767794352
        dh  = -0.2917512594
        drh = -2.76413026e-6
        ddh = -5.92994449e-6
        pxh =  0.379210
        rvh = -7.6

        r5, d5, dr5, dd5, px5, rv5 = pysofa.H2fk5(rh, dh, drh, ddh, pxh, rvh)

        self.assertAlmostEqual(r5, 1.767794455700065506, delta=1e-13)
        self.assertAlmostEqual(d5, -0.2917513626469638890, delta=1e-13)
        self.assertAlmostEqual(dr5, -0.27597945024511204e-5, delta=1e-18)
        self.assertAlmostEqual(dd5, -0.59308014093262838e-5, delta=1e-18)
        self.assertAlmostEqual(px5, 0.37921, delta=1e-13)
        self.assertAlmostEqual(rv5, -7.6000001309071126, delta=1e-10)

    def test_Hfk5z(self):
        rh =  1.767794352
        dh = -0.2917512594

        r5, d5, dr5, dd5 = pysofa.Hfk5z(rh, dh, 2400000.5, 54479.0)

        self.assertAlmostEqual(r5, 1.767794490535581026, delta=1e-13)
        self.assertAlmostEqual(d5, -0.2917513695320114258, delta=1e-14)
        self.assertAlmostEqual(dr5, 0.4335890983539243029e-8, delta=1e-22)
        self.assertAlmostEqual(dd5, -0.8569648841237745902e-9, delta=1e-23)

    def test_Ir(self):
        r = ndarray(shape=(3,3))
        r[0][0] = 2.0
        r[0][1] = 3.0
        r[0][2] = 2.0

        r[1][0] = 3.0
        r[1][1] = 2.0
        r[1][2] = 3.0

        r[2][0] = 3.0
        r[2][1] = 4.0
        r[2][2] = 5.0

        r = pysofa.Ir()

        self.assertAlmostEqual(r[0,0], 1.0, delta=0.0)
        self.assertAlmostEqual(r[0,1], 0.0, delta=0.0)
        self.assertAlmostEqual(r[0,2], 0.0, delta=0.0)

        self.assertAlmostEqual(r[1,0], 0.0, delta=0.0)
        self.assertAlmostEqual(r[1,1], 1.0, delta=0.0)
        self.assertAlmostEqual(r[1,2], 0.0, delta=0.0)

        self.assertAlmostEqual(r[2,0], 0.0, delta=0.0)
        self.assertAlmostEqual(r[2,1], 0.0, delta=0.0)
        self.assertAlmostEqual(r[2,2], 1.0, delta=0.0)

    def test_Jd2cal(self):
        dj1 = 2400000.5
        dj2 = 50123.9999

        iy, im, id, fd, j = pysofa.Jd2cal(dj1, dj2)

        self.assertEqual(iy, 1996)
        self.assertEqual(im, 2)
        self.assertEqual(id, 10)
        self.assertAlmostEqual(fd, 0.9999, 6)
        self.assertEqual(j, 0)

    def test_Jdcalf(self):
        dj1 = 2400000.5
        dj2 = 50123.9999

        iymdf, j = pysofa.Jdcalf(4, dj1, dj2)

        self.assertEqual(iymdf[0], 1996)
        self.assertEqual(iymdf[1], 2)
        self.assertEqual(iymdf[2], 10)
        self.assertEqual(iymdf[3], 9999)

        self.assertEqual(j, 0)

    def test_Num00a(self):
        rmatn = pysofa.Num00a(2400000.5, 53736.0)

        self.assertAlmostEqual(rmatn[0,0], 0.9999999999536227949, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,1], 0.8836238544090873336e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,2], 0.3830835237722400669e-5, delta=1e-12)

        self.assertAlmostEqual(rmatn[1,0], -0.8836082880798569274e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,1], 0.9999999991354655028, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,2], -0.4063240865362499850e-4, delta=1e-12)

        self.assertAlmostEqual(rmatn[2,0], -0.3831194272065995866e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,1], 0.4063237480216291775e-4, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,2], 0.9999999991671660338, delta=1e-12)

    def test_Num00b(self):
        rmatn = pysofa.Num00b(2400000.5, 53736.0)

        self.assertAlmostEqual(rmatn[0,0], 0.9999999999536069682, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,1], 0.8837746144871248011e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,2], 0.3831488838252202945e-5, delta=1e-12)

        self.assertAlmostEqual(rmatn[1,0], -0.8837590456632304720e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,1], 0.9999999991354692733, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,2], -0.4063198798559591654e-4, delta=1e-12)

        self.assertAlmostEqual(rmatn[2,0], -0.3831847930134941271e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,1], 0.4063195412258168380e-4, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,2], 0.9999999991671806225, delta=1e-12)

    def test_Num06a(self):
        rmatn = pysofa.Num06a(2400000.5, 53736)

        self.assertAlmostEqual(rmatn[0,0], 0.9999999999536227668, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,1], 0.8836241998111535233e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,2], 0.3830834608415287707e-5, delta=1e-12)

        self.assertAlmostEqual(rmatn[1,0], -0.8836086334870740138e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,1], 0.9999999991354657474, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,2], -0.4063240188248455065e-4, delta=1e-12)

        self.assertAlmostEqual(rmatn[2,0], -0.3831193642839398128e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,1], 0.4063236803101479770e-4, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,2], 0.9999999991671663114, delta=1e-12)

    def test_Numat(self):
        epsa =  0.4090789763356509900
        dpsi = -0.9630909107115582393e-5
        deps =  0.4063239174001678826e-4

        rmatn = pysofa.Numat(epsa, dpsi, deps)

        self.assertAlmostEqual(rmatn[0,0], 0.9999999999536227949, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,1], 0.8836239320236250577e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,2], 0.3830833447458251908e-5, delta=1e-12)

        self.assertAlmostEqual(rmatn[1,0], -0.8836083657016688588e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,1], 0.9999999991354654959, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,2], -0.4063240865361857698e-4, delta=1e-12)

        self.assertAlmostEqual(rmatn[2,0], -0.3831192481833385226e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,1], 0.4063237480216934159e-4, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,2], 0.9999999991671660407, delta=1e-12)

    def test_Nut00a(self):
        dpsi, deps = pysofa.Nut00a(2400000.5, 53736.0)

        self.assertAlmostEqual(dpsi, -0.9630909107115518431e-5, delta=1e-13)
        self.assertAlmostEqual(deps,  0.4063239174001678710e-4, delta=1e-13)

    def test_Nut00b(self):
        dpsi, deps = pysofa.Nut00b(2400000.5, 53736.0)

        self.assertAlmostEqual(dpsi, -0.9632552291148362783e-5, delta=1e-13)
        self.assertAlmostEqual(deps,  0.4063197106621159367e-4, delta=1e-13)

    def test_Nut06a(self):
        dpsi, deps = pysofa.Nut06a(2400000.5, 53736.0)

        self.assertAlmostEqual(dpsi, -0.9630912025820308797e-5, delta=1e-13)
        self.assertAlmostEqual(deps,  0.4063238496887249798e-4, delta=1e-13)

    def test_Nut80(self):
        dpsi, deps = pysofa.Nut80(2400000.5, 53736.0)

        self.assertAlmostEqual(dpsi, -0.9643658353226563966e-5, delta=1e-13)
        self.assertAlmostEqual(deps,  0.4060051006879713322e-4, delta=1e-13)

    def test_Nutm80(self):
        rmatn = pysofa.Nutm80(2400000.5, 53736.0)

        self.assertAlmostEqual(rmatn[0,0], 0.9999999999534999268, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,1], 0.8847935789636432161e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[0,2], 0.3835906502164019142e-5, delta=1e-12)

        self.assertAlmostEqual(rmatn[1,0], -0.8847780042583435924e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,1], 0.9999999991366569963, delta=1e-12)
        self.assertAlmostEqual(rmatn[1,2], -0.4060052702727130809e-4, delta=1e-12)

        self.assertAlmostEqual(rmatn[2,0], -0.3836265729708478796e-5, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,1], 0.4060049308612638555e-4, delta=1e-12)
        self.assertAlmostEqual(rmatn[2,2], 0.9999999991684415129, delta=1e-12)

    def test_Obl06(self):
        self.assertAlmostEqual(pysofa.Obl06(2400000.5, 54388.0),
                                0.4090749229387258204, delta=1e-14)

    def test_Obl80(self):
        self.assertAlmostEqual(pysofa.Obl80(2400000.5, 54388.0),
                                0.4090751347643816218, delta=1e-14)

    def test_P06e(self):
        eps0, psia, oma, bpa, bqa, pia, bpia, epsa, chia, za, zetaa, \
        thetaa, pa, gam, phi, psi = pysofa.P06e(2400000.5, 52541.0)

        self.assertAlmostEqual(eps0, 0.4090926006005828715, delta=1e-14)
        self.assertAlmostEqual(psia, 0.6664369630191613431e-3, delta=1e-14)
        self.assertAlmostEqual(oma , 0.4090925973783255982, delta=1e-14)
        self.assertAlmostEqual(bpa, 0.5561149371265209445e-6, delta=1e-14)
        self.assertAlmostEqual(bqa, -0.6191517193290621270e-5, delta=1e-14)
        self.assertAlmostEqual(pia, 0.6216441751884382923e-5, delta=1e-14)
        self.assertAlmostEqual(bpia, 3.052014180023779882, delta=1e-14)
        self.assertAlmostEqual(epsa, 0.4090864054922431688, delta=1e-14)
        self.assertAlmostEqual(chia, 0.1387703379530915364e-5, delta=1e-14)
        self.assertAlmostEqual(za, 0.2921789846651790546e-3, delta=1e-14)
        self.assertAlmostEqual(zetaa, 0.3178773290332009310e-3, delta=1e-14)
        self.assertAlmostEqual(thetaa, 0.2650932701657497181e-3, delta=1e-14)
        self.assertAlmostEqual(pa, 0.6651637681381016344e-3, delta=1e-14)
        self.assertAlmostEqual(gam, 0.1398077115963754987e-5, delta=1e-14)
        self.assertAlmostEqual(phi, 0.4090864090837462602, delta=1e-14)
        self.assertAlmostEqual(psi, 0.6664464807480920325e-3, delta=1e-14)

    def test_P2pv(self):
        p = ndarray(shape=(3))
        p[0] = 0.25
        p[1] = 1.2
        p[2] = 3.0

        pv = ndarray(shape=(2,3))
        pv[0][0] =  0.3
        pv[0][1] =  1.2
        pv[0][2] = -2.5

        pv[1][0] = -0.5
        pv[1][1] =  3.1
        pv[1][2] =  0.9

        pv = pysofa.P2pv(p)

        self.assertAlmostEqual(pv[0,0], 0.25, delta=0.0)
        self.assertAlmostEqual(pv[0,1], 1.2,  delta=0.0)
        self.assertAlmostEqual(pv[0,2], 3.0,  delta=0.0)

        self.assertAlmostEqual(pv[1,0], 0.0,  delta=0.0)
        self.assertAlmostEqual(pv[1,1], 0.0,  delta=0.0)
        self.assertAlmostEqual(pv[1,2], 0.0,  delta=0.0)

    def test_P2s(self):
        p = (100.0, -50.0, 25.0)

        theta, phi, r = pysofa.P2s(p)

        self.assertAlmostEqual(theta, -0.4636476090008061162, delta=1e-12)
        self.assertAlmostEqual(phi, 0.2199879773954594463, delta=1e-12)
        self.assertAlmostEqual(r, 114.5643923738960002, delta=1e-9)

    def test_Pap(self):
        a = array([1., 0.1, 0.2])
        b = (-3.0, 1e-3, 0.2)

        theta = pysofa.Pap(a, b)

        self.assertAlmostEqual(theta, 0.3671514267841113674, delta=1e-12)

    def test_Pas(self):
        al =  1.0
        ap =  0.1
        bl =  0.2
        bp = -1.0

        theta = pysofa.Pas(al, ap, bl, bp)

        self.assertAlmostEqual(theta, -2.724544922932270424, delta=1e-12)

    def test_Pb06(self):
        bzeta, bz, btheta = pysofa.Pb06(2400000.5, 50123.9999)

        self.assertAlmostEqual(bzeta, -0.5092634016326478238e-3, delta=1e-12)
        self.assertAlmostEqual(bz, -0.3602772060566044413e-3, delta=1e-12)
        self.assertAlmostEqual(btheta, -0.3779735537167811177e-3, delta=1e-12)

    def test_Pdp(self):
        a = (2.0, 2.0, 3.)
        b = (1., 3., 4.)

        adb = pysofa.Pdp(a, b)

        self.assertAlmostEqual(adb, 20, delta=1e-12)

    def test_Pfw06(self):
        gamb, phib, psib, epsa = pysofa.Pfw06(2400000.5, 50123.9999)

        self.assertAlmostEqual(gamb, -0.2243387670997995690e-5, delta=1e-16)
        self.assertAlmostEqual(phib,  0.4091014602391312808, delta=1e-12)
        self.assertAlmostEqual(psib, -0.9501954178013031895e-3, delta=1e-14)
        self.assertAlmostEqual(epsa,  0.4091014316587367491, delta=1e-12)

    def test_Plan94(self):
        pv, j = pysofa.Plan94(2400000.5, 1e6, 0)

        self.assertAlmostEqual(pv[0,0], 0.0, 1)
        self.assertAlmostEqual(pv[0,1], 0.0, 1)
        self.assertAlmostEqual(pv[0,2], 0.0, 1)

        self.assertAlmostEqual(pv[1,0], 0.0, 1)
        self.assertAlmostEqual(pv[1,1], 0.0, 1)
        self.assertAlmostEqual(pv[1,2], 0.0, 1)

        self.assertEqual(j, -1)

        pv, j = pysofa.Plan94(2400000.5, 1e6, 10)

        self.assertEqual(j, -1)

        pv, j = pysofa.Plan94(2400000.5, -320000, 3)

        self.assertAlmostEqual(pv[0,0], 0.9308038666832975759, delta=1e-11)
        self.assertAlmostEqual(pv[0,1], 0.3258319040261346000, delta=1e-11)
        self.assertAlmostEqual(pv[0,2], 0.1422794544481140560, delta=1e-11)

        self.assertAlmostEqual(pv[1,0], -0.6429458958255170006e-2, delta=1e-11)
        self.assertAlmostEqual(pv[1,1], 0.1468570657704237764e-1, delta=1e-11)
        self.assertAlmostEqual(pv[1,2], 0.6406996426270981189e-2, delta=1e-11)

        self.assertEqual(j, 1)

        pv, j = pysofa.Plan94(2400000.5, 43999.9, 1)

        self.assertAlmostEqual(pv[0,0], 0.2945293959257430832, delta=1e-11)
        self.assertAlmostEqual(pv[0,1], -0.2452204176601049596, delta=1e-11)
        self.assertAlmostEqual(pv[0,2], -0.1615427700571978153, delta=1e-11)

        self.assertAlmostEqual(pv[1,0], 0.1413867871404614441e-1, delta=1e-11)
        self.assertAlmostEqual(pv[1,1], 0.1946548301104706582e-1, delta=1e-11)
        self.assertAlmostEqual(pv[1,2], 0.8929809783898904786e-2, delta=1e-11)

        self.assertEqual(j, 0)

    def test_Pmat00(self):
        rbp = pysofa.Pmat00(2400000.5, 50123.9999)

        self.assertAlmostEqual(rbp[0,0], 0.9999995505175087260, delta=1e-12)
        self.assertAlmostEqual(rbp[0,1], 0.8695405883617884705e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[0,2], 0.3779734722239007105e-3, delta=1e-14)

        self.assertAlmostEqual(rbp[1,0], -0.8695405990410863719e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[1,1], 0.9999996219494925900, delta=1e-12)
        self.assertAlmostEqual(rbp[1,2], -0.1360775820404982209e-6, delta=1e-14)

        self.assertAlmostEqual(rbp[2,0], -0.3779734476558184991e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[2,1], -0.1925857585832024058e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,2], 0.9999999285680153377, delta=1e-12)

    def test_Pmat06(self):
        rbp = pysofa.Pmat06(2400000.5, 50123.9999)

        self.assertAlmostEqual(rbp[0,0], 0.9999995505176007047, delta=1e-12)
        self.assertAlmostEqual(rbp[0,1], 0.8695404617348208406e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[0,2], 0.3779735201865589104e-3, delta=1e-14)

        self.assertAlmostEqual(rbp[1,0], -0.8695404723772031414e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[1,1], 0.9999996219496027161, delta=1e-12)
        self.assertAlmostEqual(rbp[1,2], -0.1361752497080270143e-6, delta=1e-14)

        self.assertAlmostEqual(rbp[2,0], -0.3779734957034089490e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[2,1], -0.1924880847894457113e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,2], 0.9999999285679971958, delta=1e-14)

    def test_Pmat76(self):
        rmatp = pysofa.Pmat76(2400000.5, 50123.9999)

        self.assertAlmostEqual(rmatp[0,0], 0.9999995504328350733, delta=1e-12)
        self.assertAlmostEqual(rmatp[0,1], 0.8696632209480960785e-3, delta=1e-14)
        self.assertAlmostEqual(rmatp[0,2], 0.3779153474959888345e-3, delta=1e-14)

        self.assertAlmostEqual(rmatp[1,0], -0.8696632209485112192e-3, delta=1e-14)
        self.assertAlmostEqual(rmatp[1,1], 0.9999996218428560614, delta=1e-12)
        self.assertAlmostEqual(rmatp[1,2], -0.1643284776111886407e-6, delta=1e-14)

        self.assertAlmostEqual(rmatp[2,0], -0.3779153474950335077e-3, delta=1e-14)
        self.assertAlmostEqual(rmatp[2,1], -0.1643306746147366896e-6, delta=1e-14)
        self.assertAlmostEqual(rmatp[2,2], 0.9999999285899790119, delta=1e-12)

    def test_Pm(self):
        p = (0.3, 1.2, -2.5)

        r = pysofa.Pm(p)

        self.assertAlmostEqual(r, 2.789265136196270604, delta=1e-12)

    def test_Pmp(self):
        a = array((2., 2., 3.))
        b = (1., 3., 4.)

        amb = pysofa.Pmp(a, b)

        self.assertAlmostEqual(amb[0,0],  1.0, delta=1e-12)
        self.assertAlmostEqual(amb[0,1], -1.0, delta=1e-12)
        self.assertAlmostEqual(amb[0,2], -1.0, delta=1e-12)

    def test_Pn(self):
        p = (0.3, 1.2, -2.5)

        r, u = pysofa.Pn(p)

        self.assertAlmostEqual(r, 2.789265136196270604, delta=1e-12)

        self.assertAlmostEqual(u[0,0], 0.1075552109073112058, delta=1e-12)
        self.assertAlmostEqual(u[0,1], 0.4302208436292448232, delta=1e-12)
        self.assertAlmostEqual(u[0,2], -0.8962934242275933816, delta=1e-12)

    def test_Pn00(self):
        dpsi = -0.9632552291149335877e-5
        deps =  0.4063197106621141414e-4

        epsa, rb, rp, rbp, rn, rbpn = pysofa.Pn00(2400000.5, 53736.0, dpsi, deps)

        self.assertAlmostEqual(epsa, 0.4090791789404229916, delta=1e-12)

        self.assertAlmostEqual(rb[0,0], 0.9999999999999942498, delta=1e-12)
        self.assertAlmostEqual(rb[0,1], -0.7078279744199196626e-7, delta=1e-18)
        self.assertAlmostEqual(rb[0,2], 0.8056217146976134152e-7, delta=1e-18)

        self.assertAlmostEqual(rb[1,0], 0.7078279477857337206e-7, delta=1e-18)
        self.assertAlmostEqual(rb[1,1], 0.9999999999999969484, delta=1e-12)
        self.assertAlmostEqual(rb[1,2], 0.3306041454222136517e-7, delta=1e-18)

        self.assertAlmostEqual(rb[2,0], -0.8056217380986972157e-7, delta=1e-18)
        self.assertAlmostEqual(rb[2,1], -0.3306040883980552500e-7, delta=1e-18)
        self.assertAlmostEqual(rb[2,2], 0.9999999999999962084, delta=1e-12)

        self.assertAlmostEqual(rp[0,0], 0.9999989300532289018, delta=1e-12)
        self.assertAlmostEqual(rp[0,1], -0.1341647226791824349e-2, delta=1e-14)
        self.assertAlmostEqual(rp[0,2], -0.5829880927190296547e-3, delta=1e-14)

        self.assertAlmostEqual(rp[1,0], 0.1341647231069759008e-2, delta=1e-14)
        self.assertAlmostEqual(rp[1,1], 0.9999990999908750433, delta=1e-12)
        self.assertAlmostEqual(rp[1,2], -0.3837444441583715468e-6, delta=1e-14)

        self.assertAlmostEqual(rp[2,0], 0.5829880828740957684e-3, delta=1e-14)
        self.assertAlmostEqual(rp[2,1], -0.3984203267708834759e-6, delta=1e-14)
        self.assertAlmostEqual(rp[2,2], 0.9999998300623538046, delta=1e-12)

        self.assertAlmostEqual(rbp[0,0], 0.9999989300052243993, delta=1e-12)
        self.assertAlmostEqual(rbp[0,1], -0.1341717990239703727e-2, delta=1e-14)
        self.assertAlmostEqual(rbp[0,2], -0.5829075749891684053e-3, delta=1e-14)

        self.assertAlmostEqual(rbp[1,0], 0.1341718013831739992e-2, delta=1e-14)
        self.assertAlmostEqual(rbp[1,1], 0.9999990998959191343, delta=1e-12)
        self.assertAlmostEqual(rbp[1,2], -0.3505759733565421170e-6, delta=1e-14)

        self.assertAlmostEqual(rbp[2,0], 0.5829075206857717883e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[2,1], -0.4315219955198608970e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,2], 0.9999998301093036269, delta=1e-12)

        self.assertAlmostEqual(rn[0,0], 0.9999999999536069682, delta=1e-12)
        self.assertAlmostEqual(rn[0,1], 0.8837746144872140812e-5, delta=1e-16)
        self.assertAlmostEqual(rn[0,2], 0.3831488838252590008e-5, delta=1e-16)

        self.assertAlmostEqual(rn[1,0], -0.8837590456633197506e-5, delta=1e-16)
        self.assertAlmostEqual(rn[1,1], 0.9999999991354692733, delta=1e-12)
        self.assertAlmostEqual(rn[1,2], -0.4063198798559573702e-4, delta=1e-16)

        self.assertAlmostEqual(rn[2,0], -0.3831847930135328368e-5, delta=1e-16)
        self.assertAlmostEqual(rn[2,1], 0.4063195412258150427e-4, delta=1e-16)
        self.assertAlmostEqual(rn[2,2], 0.9999999991671806225, delta=1e-12)

        self.assertAlmostEqual(rbpn[0,0], 0.9999989440499982806, delta=1e-12)
        self.assertAlmostEqual(rbpn[0,1], -0.1332880253640848301e-2, delta=1e-14)
        self.assertAlmostEqual(rbpn[0,2], -0.5790760898731087295e-3, delta=1e-14)

        self.assertAlmostEqual(rbpn[1,0], 0.1332856746979948745e-2, delta=1e-14)
        self.assertAlmostEqual(rbpn[1,1], 0.9999991109064768883, delta=1e-12)
        self.assertAlmostEqual(rbpn[1,2], -0.4097740555723063806e-4, delta=1e-14)

        self.assertAlmostEqual(rbpn[2,0], 0.5791301929950205000e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,1], 0.4020553681373702931e-4, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,2], 0.9999998314958529887, delta=1e-12)

    def test_Pn00a(self):
        dpsi, deps, epsa, rb, rp, rbp, rn, rbpn = \
                                                pysofa.Pn00a(2400000.5, 53736.0)

        self.assertAlmostEqual(dpsi, -0.9630909107115518431e-5, delta=1e-12)
        self.assertAlmostEqual(deps,  0.4063239174001678710e-4, delta=1e-12)
        self.assertAlmostEqual(epsa,  0.4090791789404229916, delta=1e-12)

        self.assertAlmostEqual(rb[0,0], 0.9999999999999942498, delta=1e-12)
        self.assertAlmostEqual(rb[0,1], -0.7078279744199196626e-7, delta=1e-16)
        self.assertAlmostEqual(rb[0,2], 0.8056217146976134152e-7, delta=1e-16)

        self.assertAlmostEqual(rb[1,0], 0.7078279477857337206e-7, delta=1e-16)
        self.assertAlmostEqual(rb[1,1], 0.9999999999999969484, delta=1e-12)
        self.assertAlmostEqual(rb[1,2], 0.3306041454222136517e-7, delta=1e-16)

        self.assertAlmostEqual(rb[2,0], -0.8056217380986972157e-7, delta=1e-16)
        self.assertAlmostEqual(rb[2,1], -0.3306040883980552500e-7, delta=1e-16)
        self.assertAlmostEqual(rb[2,2], 0.9999999999999962084, delta=1e-12)

        self.assertAlmostEqual(rp[0,0], 0.9999989300532289018, delta=1e-12)
        self.assertAlmostEqual(rp[0,1], -0.1341647226791824349e-2, delta=1e-14)
        self.assertAlmostEqual(rp[0,2], -0.5829880927190296547e-3, delta=1e-14)

        self.assertAlmostEqual(rp[1,0], 0.1341647231069759008e-2, delta=1e-14)
        self.assertAlmostEqual(rp[1,1], 0.9999990999908750433, delta=1e-12)
        self.assertAlmostEqual(rp[1,2], -0.3837444441583715468e-6, delta=1e-14)

        self.assertAlmostEqual(rp[2,0], 0.5829880828740957684e-3, delta=1e-14)
        self.assertAlmostEqual(rp[2,1], -0.3984203267708834759e-6, delta=1e-14)
        self.assertAlmostEqual(rp[2,2], 0.9999998300623538046, delta=1e-12)

        self.assertAlmostEqual(rbp[0,0], 0.9999989300052243993, delta=1e-12)
        self.assertAlmostEqual(rbp[0,1], -0.1341717990239703727e-2, delta=1e-14)
        self.assertAlmostEqual(rbp[0,2], -0.5829075749891684053e-3, delta=1e-14)

        self.assertAlmostEqual(rbp[1,0], 0.1341718013831739992e-2, delta=1e-14)
        self.assertAlmostEqual(rbp[1,1], 0.9999990998959191343, delta=1e-12)
        self.assertAlmostEqual(rbp[1,2], -0.3505759733565421170e-6, delta=1e-14)

        self.assertAlmostEqual(rbp[2,0], 0.5829075206857717883e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[2,1], -0.4315219955198608970e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,2], 0.9999998301093036269, delta=1e-12)

        self.assertAlmostEqual(rn[0,0], 0.9999999999536227949, delta=1e-12)
        self.assertAlmostEqual(rn[0,1], 0.8836238544090873336e-5, delta=1e-14)
        self.assertAlmostEqual(rn[0,2], 0.3830835237722400669e-5, delta=1e-14)

        self.assertAlmostEqual(rn[1,0], -0.8836082880798569274e-5, delta=1e-14)
        self.assertAlmostEqual(rn[1,1], 0.9999999991354655028, delta=1e-12)
        self.assertAlmostEqual(rn[1,2], -0.4063240865362499850e-4, delta=1e-14)

        self.assertAlmostEqual(rn[2,0], -0.3831194272065995866e-5, delta=1e-14)
        self.assertAlmostEqual(rn[2,1], 0.4063237480216291775e-4, delta=1e-14)
        self.assertAlmostEqual(rn[2,2], 0.9999999991671660338, delta=1e-12)

        self.assertAlmostEqual(rbpn[0,0], 0.9999989440476103435, delta=1e-12)
        self.assertAlmostEqual(rbpn[0,1], -0.1332881761240011763e-2, delta=1e-14)
        self.assertAlmostEqual(rbpn[0,2], -0.5790767434730085751e-3, delta=1e-14)

        self.assertAlmostEqual(rbpn[1,0], 0.1332858254308954658e-2, delta=1e-14)
        self.assertAlmostEqual(rbpn[1,1], 0.9999991109044505577, delta=1e-12)
        self.assertAlmostEqual(rbpn[1,2], -0.4097782710396580452e-4, delta=1e-14)

        self.assertAlmostEqual(rbpn[2,0], 0.5791308472168152904e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,1], 0.4020595661591500259e-4, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,2], 0.9999998314954572304, delta=1e-12)

    def test_Pn00b(self):
        dpsi, deps, epsa, rb, rp, rbp, rn, rbpn = \
                                                pysofa.Pn00b(2400000.5, 53736.0)

        self.assertAlmostEqual(dpsi, -0.9632552291148362783e-5, delta=1e-12)
        self.assertAlmostEqual(deps,  0.4063197106621159367e-4, delta=1e-12)
        self.assertAlmostEqual(epsa,  0.4090791789404229916, delta=1e-12)

        self.assertAlmostEqual(rb[0,0], 0.9999999999999942498, delta=1e-12)
        self.assertAlmostEqual(rb[0,1], -0.7078279744199196626e-7, delta=1e-16)
        self.assertAlmostEqual(rb[0,2], 0.8056217146976134152e-7, delta=1e-16)

        self.assertAlmostEqual(rb[1,0], 0.7078279477857337206e-7, delta=1e-16)
        self.assertAlmostEqual(rb[1,1], 0.9999999999999969484, delta=1e-12)
        self.assertAlmostEqual(rb[1,2], 0.3306041454222136517e-7, delta=1e-16)

        self.assertAlmostEqual(rb[2,0], -0.8056217380986972157e-7, delta=1e-16)
        self.assertAlmostEqual(rb[2,1], -0.3306040883980552500e-7, delta=1e-16)
        self.assertAlmostEqual(rb[2,2], 0.9999999999999962084, delta=1e-12)

        self.assertAlmostEqual(rp[0,0], 0.9999989300532289018, delta=1e-12)
        self.assertAlmostEqual(rp[0,1], -0.1341647226791824349e-2, delta=1e-14)
        self.assertAlmostEqual(rp[0,2], -0.5829880927190296547e-3, delta=1e-14)

        self.assertAlmostEqual(rp[1,0], 0.1341647231069759008e-2, delta=1e-14)
        self.assertAlmostEqual(rp[1,1], 0.9999990999908750433, delta=1e-12)
        self.assertAlmostEqual(rp[1,2], -0.3837444441583715468e-6, delta=1e-14)

        self.assertAlmostEqual(rp[2,0], 0.5829880828740957684e-3, delta=1e-14)
        self.assertAlmostEqual(rp[2,1], -0.3984203267708834759e-6, delta=1e-14)
        self.assertAlmostEqual(rp[2,2], 0.9999998300623538046, delta=1e-12)

        self.assertAlmostEqual(rbp[0,0], 0.9999989300052243993, delta=1e-12)
        self.assertAlmostEqual(rbp[0,1], -0.1341717990239703727e-2, delta=1e-14)
        self.assertAlmostEqual(rbp[0,2], -0.5829075749891684053e-3, delta=1e-14)

        self.assertAlmostEqual(rbp[1,0], 0.1341718013831739992e-2, delta=1e-14)
        self.assertAlmostEqual(rbp[1,1], 0.9999990998959191343, delta=1e-12)
        self.assertAlmostEqual(rbp[1,2], -0.3505759733565421170e-6, delta=1e-14)

        self.assertAlmostEqual(rbp[2,0], 0.5829075206857717883e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[2,1], -0.4315219955198608970e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,2], 0.9999998301093036269, delta=1e-12)

        self.assertAlmostEqual(rn[0,0], 0.9999999999536069682, delta=1e-12)
        self.assertAlmostEqual(rn[0,1], 0.8837746144871248011e-5, delta=1e-14)
        self.assertAlmostEqual(rn[0,2], 0.3831488838252202945e-5, delta=1e-14)

        self.assertAlmostEqual(rn[1,0], -0.8837590456632304720e-5, delta=1e-14)
        self.assertAlmostEqual(rn[1,1], 0.9999999991354692733, delta=1e-12)
        self.assertAlmostEqual(rn[1,2], -0.4063198798559591654e-4, delta=1e-14)

        self.assertAlmostEqual(rn[2,0], -0.3831847930134941271e-5, delta=1e-14)
        self.assertAlmostEqual(rn[2,1], 0.4063195412258168380e-4, delta=1e-14)
        self.assertAlmostEqual(rn[2,2], 0.9999999991671806225, delta=1e-12)

        self.assertAlmostEqual(rbpn[0,0], 0.9999989440499982806, delta=1e-12)
        self.assertAlmostEqual(rbpn[0,1], -0.1332880253640849194e-2, delta=1e-14)
        self.assertAlmostEqual(rbpn[0,2], -0.5790760898731091166e-3, delta=1e-14)

        self.assertAlmostEqual(rbpn[1,0], 0.1332856746979949638e-2, delta=1e-14)
        self.assertAlmostEqual(rbpn[1,1], 0.9999991109064768883, delta=1e-12)
        self.assertAlmostEqual(rbpn[1,2], -0.4097740555723081811e-4, delta=1e-14)

        self.assertAlmostEqual(rbpn[2,0], 0.5791301929950208873e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,1], 0.4020553681373720832e-4, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,2], 0.9999998314958529887, delta=1e-12)

    def test_Pn06a(self):
        dpsi, deps, epsa, rb, rp, rbp, rn, rbpn = \
                                                pysofa.Pn06a(2400000.5, 53736.0)

        self.assertAlmostEqual(dpsi, -0.9630912025820308797e-5, delta=1e-12)
        self.assertAlmostEqual(deps,  0.4063238496887249798e-4, delta=1e-12)
        self.assertAlmostEqual(epsa,  0.4090789763356509926, delta=1e-12)

        self.assertAlmostEqual(rb[0,0], 0.9999999999999942497, delta=1e-12)
        self.assertAlmostEqual(rb[0,1], -0.7078368960971557145e-7, delta=1e-14)
        self.assertAlmostEqual(rb[0,2], 0.8056213977613185606e-7, delta=1e-14)

        self.assertAlmostEqual(rb[1,0], 0.7078368694637674333e-7, delta=1e-14)
        self.assertAlmostEqual(rb[1,1], 0.9999999999999969484, delta=1e-12)
        self.assertAlmostEqual(rb[1,2], 0.3305943742989134124e-7, delta=1e-14)

        self.assertAlmostEqual(rb[2,0], -0.8056214211620056792e-7, delta=1e-14)
        self.assertAlmostEqual(rb[2,1], -0.3305943172740586950e-7, delta=1e-14)
        self.assertAlmostEqual(rb[2,2], 0.9999999999999962084, delta=1e-12)

        self.assertAlmostEqual(rp[0,0], 0.9999989300536854831, delta=1e-12)
        self.assertAlmostEqual(rp[0,1], -0.1341646886204443795e-2, delta=1e-14)
        self.assertAlmostEqual(rp[0,2], -0.5829880933488627759e-3, delta=1e-14)

        self.assertAlmostEqual(rp[1,0], 0.1341646890569782183e-2, delta=1e-14)
        self.assertAlmostEqual(rp[1,1], 0.9999990999913319321, delta=1e-12)
        self.assertAlmostEqual(rp[1,2], -0.3835944216374477457e-6, delta=1e-14)

        self.assertAlmostEqual(rp[2,0], 0.5829880833027867368e-3, delta=1e-14)
        self.assertAlmostEqual(rp[2,1], -0.3985701514686976112e-6, delta=1e-14)
        self.assertAlmostEqual(rp[2,2], 0.9999998300623534950, delta=1e-12)

        self.assertAlmostEqual(rbp[0,0], 0.9999989300056797893, delta=1e-12)
        self.assertAlmostEqual(rbp[0,1], -0.1341717650545059598e-2, delta=1e-14)
        self.assertAlmostEqual(rbp[0,2], -0.5829075756493728856e-3, delta=1e-14)

        self.assertAlmostEqual(rbp[1,0], 0.1341717674223918101e-2, delta=1e-14)
        self.assertAlmostEqual(rbp[1,1], 0.9999990998963748448, delta=1e-12)
        self.assertAlmostEqual(rbp[1,2], -0.3504269280170069029e-6, delta=1e-14)

        self.assertAlmostEqual(rbp[2,0], 0.5829075211461454599e-3, delta=1e-14)
        self.assertAlmostEqual(rbp[2,1], -0.4316708436255949093e-6, delta=1e-14)
        self.assertAlmostEqual(rbp[2,2], 0.9999998301093032943, delta=1e-12)

        self.assertAlmostEqual(rn[0,0], 0.9999999999536227668, delta=1e-12)
        self.assertAlmostEqual(rn[0,1], 0.8836241998111535233e-5, delta=1e-14)
        self.assertAlmostEqual(rn[0,2], 0.3830834608415287707e-5, delta=1e-14)

        self.assertAlmostEqual(rn[1,0], -0.8836086334870740138e-5, delta=1e-14)
        self.assertAlmostEqual(rn[1,1], 0.9999999991354657474, delta=1e-12)
        self.assertAlmostEqual(rn[1,2], -0.4063240188248455065e-4, delta=1e-14)

        self.assertAlmostEqual(rn[2,0], -0.3831193642839398128e-5, delta=1e-14)
        self.assertAlmostEqual(rn[2,1], 0.4063236803101479770e-4, delta=1e-14)
        self.assertAlmostEqual(rn[2,2], 0.9999999991671663114, delta=1e-12)

        self.assertAlmostEqual(rbpn[0,0], 0.9999989440480669738, delta=1e-12)
        self.assertAlmostEqual(rbpn[0,1], -0.1332881418091915973e-2, delta=1e-14)
        self.assertAlmostEqual(rbpn[0,2], -0.5790767447612042565e-3, delta=1e-14)

        self.assertAlmostEqual(rbpn[1,0], 0.1332857911250989133e-2, delta=1e-14)
        self.assertAlmostEqual(rbpn[1,1], 0.9999991109049141908, delta=1e-12)
        self.assertAlmostEqual(rbpn[1,2], -0.4097767128546784878e-4, delta=1e-14)

        self.assertAlmostEqual(rbpn[2,0], 0.5791308482835292617e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,1], 0.4020580099454020310e-4, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,2], 0.9999998314954628695, delta=1e-12)

    def test_Pnm00a(self):
        rbpn = pysofa.Pnm00a(2400000.5, 50123.9999)

        self.assertAlmostEqual(rbpn[0,0], 0.9999995832793134257, delta=1e-12)
        self.assertAlmostEqual(rbpn[0,1], 0.8372384254137809439e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[0,2], 0.3639684306407150645e-3, delta=1e-14)

        self.assertAlmostEqual(rbpn[1,0], -0.8372535226570394543e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[1,1], 0.9999996486491582471, delta=1e-12)
        self.assertAlmostEqual(rbpn[1,2], 0.4132915262664072381e-4, delta=1e-14)

        self.assertAlmostEqual(rbpn[2,0], -0.3639337004054317729e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,1], -0.4163386925461775873e-4, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,2], 0.9999999329094390695, delta=1e-12)

    def test_Pnm00b(self):
        rbpn = pysofa.Pnm00b(2400000.5, 50123.9999)

        self.assertAlmostEqual(rbpn[0,0], 0.9999995832776208280, delta=1e-12)
        self.assertAlmostEqual(rbpn[0,1], 0.8372401264429654837e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[0,2], 0.3639691681450271771e-3, delta=1e-14)

        self.assertAlmostEqual(rbpn[1,0], -0.8372552234147137424e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[1,1], 0.9999996486477686123, delta=1e-12)
        self.assertAlmostEqual(rbpn[1,2], 0.4132832190946052890e-4, delta=1e-14)

        self.assertAlmostEqual(rbpn[2,0], -0.3639344385341866407e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,1], -0.4163303977421522785e-4, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,2], 0.9999999329092049734, delta=1e-12)

    def test_Pnm06a(self):
        rbpn = pysofa.Pnm06a(2400000.5, 50123.9999)

        self.assertAlmostEqual(rbpn[0,0], 0.9999995832794205484, delta=1e-12)
        self.assertAlmostEqual(rbpn[0,1], 0.8372382772630962111e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[0,2], 0.3639684771140623099e-3, delta=1e-14)

        self.assertAlmostEqual(rbpn[1,0], -0.8372533744743683605e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[1,1], 0.9999996486492861646, delta=1e-12)
        self.assertAlmostEqual(rbpn[1,2], 0.4132905944611019498e-4, delta=1e-14)

        self.assertAlmostEqual(rbpn[2,0], -0.3639337469629464969e-3, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,1], -0.4163377605910663999e-4, delta=1e-14)
        self.assertAlmostEqual(rbpn[2,2], 0.9999999329094260057, delta=1e-12)

    def test_Pnm80(self):
        rmatpn = pysofa.Pnm80(2400000.5, 50123.9999)

        self.assertAlmostEqual(rmatpn[0,0], 0.9999995831934611169, delta=1e-12)
        self.assertAlmostEqual(rmatpn[0,1], 0.8373654045728124011e-3, delta=1e-14)
        self.assertAlmostEqual(rmatpn[0,2], 0.3639121916933106191e-3, delta=1e-14)

        self.assertAlmostEqual(rmatpn[1,0], -0.8373804896118301316e-3, delta=1e-14)
        self.assertAlmostEqual(rmatpn[1,1], 0.9999996485439674092, delta=1e-12)
        self.assertAlmostEqual(rmatpn[1,2], 0.4130202510421549752e-4, delta=1e-14)

        self.assertAlmostEqual(rmatpn[2,0], -0.3638774789072144473e-3, delta=1e-14)
        self.assertAlmostEqual(rmatpn[2,1], -0.4160674085851722359e-4, delta=1e-14)
        self.assertAlmostEqual(rmatpn[2,2], 0.9999999329310274805, delta=1e-12)

    def test_Pom00(self):
        xp =  2.55060238e-7
        yp =  1.860359247e-6
        sp = -0.1367174580728891460e-10

        rpom = pysofa.Pom00(xp, yp, sp)

        self.assertAlmostEqual(rpom[0,0], 0.9999999999999674721, delta=1e-12)
        self.assertAlmostEqual(rpom[0,1], -0.1367174580728846989e-10, delta=1e-16)
        self.assertAlmostEqual(rpom[0,2], 0.2550602379999972345e-6, delta=1e-16)

        self.assertAlmostEqual(rpom[1,0], 0.1414624947957029801e-10, delta=1e-16)
        self.assertAlmostEqual(rpom[1,1], 0.9999999999982695317, delta=1e-12)
        self.assertAlmostEqual(rpom[1,2], -0.1860359246998866389e-5, delta=1e-16)

        self.assertAlmostEqual(rpom[2,0], -0.2550602379741215021e-6, delta=1e-16)
        self.assertAlmostEqual(rpom[2,1], 0.1860359247002414021e-5, delta=1e-16)
        self.assertAlmostEqual(rpom[2,2], 0.9999999999982370039, delta=1e-12)

    def test_Ppp(self):
        a = (2., 2., 3.)
        b = [1., 3., 4.]

        apb = pysofa.Ppp(a, b)

        self.assertAlmostEqual(apb[0,0], 3.0, delta=1e-12)
        self.assertAlmostEqual(apb[0,1], 5.0, delta=1e-12)
        self.assertAlmostEqual(apb[0,2], 7.0, delta=1e-12)

    def test_Ppsp(self):
        a = array((2., 2., 3.))
        s = 5.0
        b = (1., 3.0, 4.0)

        apsb = pysofa.Ppsp(a, s, b)

        self.assertAlmostEqual(apsb[0,0], 7.0, delta=1e-12)
        self.assertAlmostEqual(apsb[0,1], 17.0, delta=1e-12)
        self.assertAlmostEqual(apsb[0,2], 23.0, delta=1e-12)

    def test_Pr00(self):
        dpsipr, depspr = pysofa.Pr00(2400000.5, 53736)

        self.assertAlmostEqual(dpsipr, -0.8716465172668347629e-7, delta=1e-22)
        self.assertAlmostEqual(depspr, -0.7342018386722813087e-8, delta=1e-22)

    def test_Prec76(self):
        ep01 = 2400000.5
        ep02 = 33282.0
        ep11 = 2400000.5
        ep12 = 51544.0

        zeta, z, theta = pysofa.Prec76(ep01, ep02, ep11, ep12)

        self.assertAlmostEqual(zeta,  0.5588961642000161243e-2, delta=1e-12)
        self.assertAlmostEqual(z,     0.5589922365870680624e-2, delta=1e-12)
        self.assertAlmostEqual(theta, 0.4858945471687296760e-2, delta=1e-12)

    def test_Pv2p(self):
        pv = array(((0.3, 1.2, -2.5), (-0.5, 3.1, 0.9)))

        p = pysofa.Pv2p(pv)

        self.assertAlmostEqual(p[0,0],  0.3, delta=0.0)
        self.assertAlmostEqual(p[0,1],  1.2, delta=0.0)
        self.assertAlmostEqual(p[0,2], -2.5, delta=0.0)

    def test_Pv2s(self):
        pv = ndarray(shape=(2,3))
        pv[0][0] = -0.4514964673880165
        pv[0][1] =  0.03093394277342585
        pv[0][2] =  0.05594668105108779

        pv[1][0] =  1.292270850663260e-5
        pv[1][1] =  2.652814182060692e-6
        pv[1][2] =  2.568431853930293e-6

        theta, phi, r, td, pd, rd = pysofa.Pv2s(pv)

        self.assertAlmostEqual(theta, 3.073185307179586515, delta=1e-12)
        self.assertAlmostEqual(phi, 0.1229999999999999992, delta=1e-12)
        self.assertAlmostEqual(r, 0.4559999999999999757, delta=1e-12)
        self.assertAlmostEqual(td, -0.7800000000000000364e-5, delta=1e-16)
        self.assertAlmostEqual(pd, 0.9010000000000001639e-5, delta=1e-16)
        self.assertAlmostEqual(rd, -0.1229999999999999832e-4, delta=1e-16)

    def test_Pvdpv(self):
        a = ndarray(shape=(2,3))
        a[0][0] = 2.0
        a[0][1] = 2.0
        a[0][2] = 3.0

        a[1][0] = 6.0
        a[1][1] = 0.0
        a[1][2] = 4.0

        b = ndarray(shape=(2,3))
        b[0][0] = 1.0
        b[0][1] = 3.0
        b[0][2] = 4.0

        b[1][0] = 0.0
        b[1][1] = 2.0
        b[1][2] = 8.0

        adb = pysofa.Pvdpv(a, b)

        self.assertAlmostEqual(adb[0,0], 20.0, delta=1e-12)
        self.assertAlmostEqual(adb[0,1], 50.0, delta=1e-12)

    def test_Pvm(self):
        pv = ndarray(shape=(2,3))
        pv[0][0] =  0.3
        pv[0][1] =  1.2
        pv[0][2] = -2.5

        pv[1][0] =  0.45
        pv[1][1] = -0.25
        pv[1][2] =  1.1

        r, s = pysofa.Pvm(pv)

        self.assertAlmostEqual(r, 2.789265136196270604, delta=1e-12)
        self.assertAlmostEqual(s, 1.214495780149111922, delta=1e-12)

    def test_Pvmpv(self):
        a = ndarray(shape=(2,3))
        a[0][0] = 2.0
        a[0][1] = 2.0
        a[0][2] = 3.0

        a[1][0] = 5.0
        a[1][1] = 6.0
        a[1][2] = 3.0

        b = ndarray(shape=(2,3))
        b[0][0] = 1.0
        b[0][1] = 3.0
        b[0][2] = 4.0

        b[1][0] = 3.0
        b[1][1] = 2.0
        b[1][2] = 1.0

        amb = pysofa.Pvmpv(a, b)

        self.assertAlmostEqual(amb[0,0],  1.0, delta=1e-12)
        self.assertAlmostEqual(amb[0,1], -1.0, delta=1e-12)
        self.assertAlmostEqual(amb[0,2], -1.0, delta=1e-12)

        self.assertAlmostEqual(amb[1,0],  2.0, delta=1e-12)
        self.assertAlmostEqual(amb[1,1],  4.0, delta=1e-12)
        self.assertAlmostEqual(amb[1,2],  2.0, delta=1e-12)

    def test_Pvppv(self):
        a = ndarray(shape=(2,3))
        a[0][0] = 2.0
        a[0][1] = 2.0
        a[0][2] = 3.0

        a[1][0] = 5.0
        a[1][1] = 6.0
        a[1][2] = 3.0

        b = ndarray(shape=(2,3))
        b[0][0] = 1.0
        b[0][1] = 3.0
        b[0][2] = 4.0

        b[1][0] = 3.0
        b[1][1] = 2.0
        b[1][2] = 1.0

        apb = pysofa.Pvppv(a, b)

        self.assertAlmostEqual(apb[0,0], 3.0, delta=1e-12)
        self.assertAlmostEqual(apb[0,1], 5.0, delta=1e-12)
        self.assertAlmostEqual(apb[0,2], 7.0, delta=1e-12)

        self.assertAlmostEqual(apb[1,0], 8.0, delta=1e-12)
        self.assertAlmostEqual(apb[1,1], 8.0, delta=1e-12)
        self.assertAlmostEqual(apb[1,2], 4.0, delta=1e-12)

    def test_Pvstar(self):
        pv = ndarray(shape=(2,3))
        pv[0][0] =  126668.5912743160601
        pv[0][1] =  2136.792716839935195
        pv[0][2] = -245251.2339876830091

        pv[1][0] = -0.4051854035740712739e-2
        pv[1][1] = -0.6253919754866173866e-2
        pv[1][2] =  0.1189353719774107189e-1

        ra, dec, pmr, pmd, px, rv, j = pysofa.Pvstar(pv)

        self.assertAlmostEqual(ra, 0.1686756e-1, delta=1e-12)
        self.assertAlmostEqual(dec, -1.093989828, delta=1e-12)
        self.assertAlmostEqual(pmr, -0.178323516e-4, delta=1e-16)
        self.assertAlmostEqual(pmd, 0.2336024047e-5, delta=1e-16)
        self.assertAlmostEqual(px, 0.74723, delta=1e-12)
        self.assertAlmostEqual(rv, -21.6, delta=1e-11)

        self.assertEqual(j, 0)

    def test_Pvu(self):
        pv = ndarray(shape=(2,3))
        pv[0][0] =  126668.5912743160734
        pv[0][1] =  2136.792716839935565
        pv[0][2] = -245251.2339876830229

        pv[1][0] = -0.4051854035740713039e-2
        pv[1][1] = -0.6253919754866175788e-2
        pv[1][2] =  0.1189353719774107615e-1

        upv = pysofa.Pvu(2920.0, pv)

        self.assertAlmostEqual(upv[0,0], 126656.7598605317105, delta=1e-12)
        self.assertAlmostEqual(upv[0,1], 2118.531271155726332, delta=1e-12)
        self.assertAlmostEqual(upv[0,2], -245216.5048590656190, delta=1e-12)

        self.assertAlmostEqual(upv[1,0], -0.4051854035740713039e-2, delta=1e-12)
        self.assertAlmostEqual(upv[1,1], -0.6253919754866175788e-2, delta=1e-12)
        self.assertAlmostEqual(upv[1,2], 0.1189353719774107615e-1, delta=1e-12)

    def test_Pvup(self):
        pv = ndarray(shape=(2,3))
        pv[0][0] =  126668.5912743160734
        pv[0][1] =  2136.792716839935565
        pv[0][2] = -245251.2339876830229

        pv[1][0] = -0.4051854035740713039e-2
        pv[1][1] = -0.6253919754866175788e-2
        pv[1][2] =  0.1189353719774107615e-1

        p = pysofa.Pvup(2920.0, pv)

        self.assertAlmostEqual(p[0,0],  126656.7598605317105,   delta=1e-12)
        self.assertAlmostEqual(p[0,1],    2118.531271155726332, delta=1e-12)
        self.assertAlmostEqual(p[0,2], -245216.5048590656190,   delta=1e-12)

    def test_Pvxpv(self):
        a = ndarray(shape=(2,3))
        a[0][0] = 2.0
        a[0][1] = 2.0
        a[0][2] = 3.0

        a[1][0] = 6.0
        a[1][1] = 0.0
        a[1][2] = 4.0

        b = ndarray(shape=(2,3))
        b[0][0] = 1.0
        b[0][1] = 3.0
        b[0][2] = 4.0

        b[1][0] = 0.0
        b[1][1] = 2.0
        b[1][2] = 8.0

        axb = pysofa.Pvxpv(a, b)

        self.assertAlmostEqual(axb[0,0],  -1.0, delta=1e-12)
        self.assertAlmostEqual(axb[0,1],  -5.0, delta=1e-12)
        self.assertAlmostEqual(axb[0,2],   4.0, delta=1e-12)

        self.assertAlmostEqual(axb[1,0],  -2.0, delta=1e-12)
        self.assertAlmostEqual(axb[1,1], -36.0, delta=1e-12)
        self.assertAlmostEqual(axb[1,2],  22.0, delta=1e-12)

    def test_Pxp(self):
        a = ndarray(shape=(3))
        a[0] = 2.0
        a[1] = 2.0
        a[2] = 3.0

        b = (1., 3., 4.)

        axb = pysofa.Pxp(a, b)

        self.assertAlmostEqual(axb[0,0], -1.0, delta=1e-12)
        self.assertAlmostEqual(axb[0,1], -5.0, delta=1e-12)
        self.assertAlmostEqual(axb[0,2],  4.0, delta=1e-12)

    def test_Rm2v(self):
        r = ndarray(shape=(3,3))
        r[0][0] =  0.00
        r[0][1] = -0.80
        r[0][2] = -0.60

        r[1][0] =  0.80
        r[1][1] = -0.36
        r[1][2] =  0.48

        r[2][0] =  0.60
        r[2][1] =  0.48
        r[2][2] = -0.64

        w = pysofa.Rm2v(r)

        self.assertAlmostEqual(w[0,0],  0.0,                  delta=1e-12)
        self.assertAlmostEqual(w[0,1],  1.413716694115406957, delta=1e-12)
        self.assertAlmostEqual(w[0,2], -1.884955592153875943, delta=1e-12)

    def test_Rv2m(self):
        w = (0, 1.41371669, -1.88495559)

        r = pysofa.Rv2m(w)

        self.assertAlmostEqual(r[0,0], -0.7071067782221119905, delta=1e-14)
        self.assertAlmostEqual(r[0,1], -0.5656854276809129651, delta=1e-14)
        self.assertAlmostEqual(r[0,2], -0.4242640700104211225, delta=1e-14)

        self.assertAlmostEqual(r[1,0],  0.5656854276809129651, delta=1e-14)
        self.assertAlmostEqual(r[1,1], -0.0925483394532274246, delta=1e-14)
        self.assertAlmostEqual(r[1,2], -0.8194112531408833269, delta=1e-14)

        self.assertAlmostEqual(r[2,0],  0.4242640700104211225, delta=1e-14)
        self.assertAlmostEqual(r[2,1], -0.8194112531408833269, delta=1e-14)
        self.assertAlmostEqual(r[2,2],  0.3854415612311154341, delta=1e-14)

    def test_Rx(self):
        phi = 0.3456789

        r = ndarray(shape=(3,3))
        r[0][0] = 2.0
        r[0][1] = 3.0
        r[0][2] = 2.0

        r[1][0] = 3.0
        r[1][1] = 2.0
        r[1][2] = 3.0

        r[2][0] = 3.0
        r[2][1] = 4.0
        r[2][2] = 5.0

        r = pysofa.Rx(phi, r)

        self.assertAlmostEqual(r[0,0], 2.0, delta=0.0)
        self.assertAlmostEqual(r[0,1], 3.0, delta=0.0)
        self.assertAlmostEqual(r[0,2], 2.0, delta=0.0)

        self.assertAlmostEqual(r[1,0], 3.839043388235612460, delta=1e-12)
        self.assertAlmostEqual(r[1,1], 3.237033249594111899, delta=1e-12)
        self.assertAlmostEqual(r[1,2], 4.516714379005982719, delta=1e-12)

        self.assertAlmostEqual(r[2,0], 1.806030415924501684, delta=1e-12)
        self.assertAlmostEqual(r[2,1], 3.085711545336372503, delta=1e-12)
        self.assertAlmostEqual(r[2,2], 3.687721683977873065, delta=1e-12)

    def test_Rxp(self):
        r = ndarray(shape=(3,3))
        r[0][0] = 2.0
        r[0][1] = 3.0
        r[0][2] = 2.0

        r[1][0] = 3.0
        r[1][1] = 2.0
        r[1][2] = 3.0

        r[2][0] = 3.0
        r[2][1] = 4.0
        r[2][2] = 5.0

        p = (.2, 1.5, .1)

        rp = pysofa.Rxp(r, p)

        self.assertAlmostEqual(rp[0,0], 5.1, delta=1e-12)
        self.assertAlmostEqual(rp[0,1], 3.9, delta=1e-12)
        self.assertAlmostEqual(rp[0,2], 7.1, delta=1e-12)

    def test_Rxpv(self):
        r = ndarray(shape=(3,3))
        r[0][0] = 2.0
        r[0][1] = 3.0
        r[0][2] = 2.0

        r[1][0] = 3.0
        r[1][1] = 2.0
        r[1][2] = 3.0

        r[2][0] = 3.0
        r[2][1] = 4.0
        r[2][2] = 5.0

        pv = ndarray(shape=(2,3))
        pv[0][0] = 0.2
        pv[0][1] = 1.5
        pv[0][2] = 0.1

        pv[1][0] = 1.5
        pv[1][1] = 0.2
        pv[1][2] = 0.1

        rpv = pysofa.Rxpv(r, pv)

        self.assertAlmostEqual(rpv[0,0], 5.1, delta=1e-12)
        self.assertAlmostEqual(rpv[1,0], 3.8, delta=1e-12)

        self.assertAlmostEqual(rpv[0,1], 3.9, delta=1e-12)
        self.assertAlmostEqual(rpv[1,1], 5.2, delta=1e-12)

        self.assertAlmostEqual(rpv[0,2], 7.1, delta=1e-12)
        self.assertAlmostEqual(rpv[1,2], 5.8, delta=1e-12)

    def test_Rxr(self):
        a = ndarray(shape=(3,3))
        a[0][0] = 2.0
        a[0][1] = 3.0
        a[0][2] = 2.0

        a[1][0] = 3.0
        a[1][1] = 2.0
        a[1][2] = 3.0

        a[2][0] = 3.0
        a[2][1] = 4.0
        a[2][2] = 5.0

        b = ndarray(shape=(3,3))
        b[0][0] = 1.0
        b[0][1] = 2.0
        b[0][2] = 2.0

        b[1][0] = 4.0
        b[1][1] = 1.0
        b[1][2] = 1.0

        b[2][0] = 3.0
        b[2][1] = 0.0
        b[2][2] = 1.0

        atb = pysofa.Rxr(a, b)

        self.assertAlmostEqual(atb[0,0], 20.0, delta=1e-12)
        self.assertAlmostEqual(atb[0,1],  7.0, delta=1e-12)
        self.assertAlmostEqual(atb[0,2],  9.0, delta=1e-12)

        self.assertAlmostEqual(atb[1,0], 20.0, delta=1e-12)
        self.assertAlmostEqual(atb[1,1],  8.0, delta=1e-12)
        self.assertAlmostEqual(atb[1,2], 11.0, delta=1e-12)

        self.assertAlmostEqual(atb[2,0], 34.0, delta=1e-12)
        self.assertAlmostEqual(atb[2,1], 10.0, delta=1e-12)
        self.assertAlmostEqual(atb[2,2], 15.0, delta=1e-12)

    def test_Ry(self):
        theta = 0.3456789

        r = ndarray(shape=(3,3))
        r[0][0] = 2.0
        r[0][1] = 3.0
        r[0][2] = 2.0

        r[1][0] = 3.0
        r[1][1] = 2.0
        r[1][2] = 3.0

        r[2][0] = 3.0
        r[2][1] = 4.0
        r[2][2] = 5.0

        r = pysofa.Ry(theta, r)

        self.assertAlmostEqual(r[0,0], 0.8651847818978159930, delta=1e-12)
        self.assertAlmostEqual(r[0,1], 1.467194920539316554, delta=1e-12)
        self.assertAlmostEqual(r[0,2], 0.1875137911274457342, delta=1e-12)

        self.assertAlmostEqual(r[1,0], 3, delta=1e-12)
        self.assertAlmostEqual(r[1,1], 2, delta=1e-12)
        self.assertAlmostEqual(r[1,2], 3, delta=1e-12)

        self.assertAlmostEqual(r[2,0], 3.500207892850427330, delta=1e-12)
        self.assertAlmostEqual(r[2,1], 4.779889022262298150, delta=1e-12)
        self.assertAlmostEqual(r[2,2], 5.381899160903798712, delta=1e-12)

    def test_Rz(self):
        psi = 0.3456789

        r = ndarray(shape=(3,3))
        r[0][0] = 2.0
        r[0][1] = 3.0
        r[0][2] = 2.0

        r[1][0] = 3.0
        r[1][1] = 2.0
        r[1][2] = 3.0

        r[2][0] = 3.0
        r[2][1] = 4.0
        r[2][2] = 5.0

        r = pysofa.Rz(psi, r)

        self.assertAlmostEqual(r[0,0], 2.898197754208926769, delta=1e-12)
        self.assertAlmostEqual(r[0,1], 3.500207892850427330, delta=1e-12)
        self.assertAlmostEqual(r[0,2], 2.898197754208926769, delta=1e-12)

        self.assertAlmostEqual(r[1,0], 2.144865911309686813, delta=1e-12)
        self.assertAlmostEqual(r[1,1], 0.865184781897815993, delta=1e-12)
        self.assertAlmostEqual(r[1,2], 2.144865911309686813, delta=1e-12)

        self.assertAlmostEqual(r[2,0], 3.0, delta=1e-12)
        self.assertAlmostEqual(r[2,1], 4.0, delta=1e-12)
        self.assertAlmostEqual(r[2,2], 5.0, delta=1e-12)

    def test_S00(self):
        x = 0.5791308486706011000e-3
        y = 0.4020579816732961219e-4

        s = pysofa.S00(2400000.5, 53736.0, x, y)

        self.assertAlmostEqual(s, -0.1220036263270905693e-7, delta=1e-18)

    def test_S00a(self):
        s = pysofa.S00a(2400000.5, 52541.0)

        self.assertAlmostEqual(s, -0.1340684448919163584e-7, delta=1e-18)

    def test_S00b(self):
        s = pysofa.S00b(2400000.5, 52541.0)

        self.assertAlmostEqual(s, -0.1340695782951026584e-7, delta=1e-18)

    def test_S06(self):
        x = 0.5791308486706011000e-3
        y = 0.4020579816732961219e-4

        s = pysofa.S06(2400000.5, 53736.0, x, y)

        self.assertAlmostEqual(s, -0.1220032213076463117e-7, delta=1e-18)

    def test_S06a(self):
        s = pysofa.S06a(2400000.5, 52541.0)

        self.assertAlmostEqual(s, -0.1340680437291812383e-7, delta=1e-18)

    def test_S2c(self):
        c = pysofa.S2c(3.0123, -0.999)

        self.assertAlmostEqual(c[0,0], -0.5366267667260523906, delta=1e-12)
        self.assertAlmostEqual(c[0,1],  0.0697711109765145365, delta=1e-12)
        self.assertAlmostEqual(c[0,2], -0.8409302618566214041, delta=1e-12)

    def test_S2p(self):
        p = pysofa.S2p(-3.21, 0.123, 0.456)

        self.assertAlmostEqual(p[0,0], -0.4514964673880165228, delta=1e-12)
        self.assertAlmostEqual(p[0,1],  0.0309339427734258688, delta=1e-12)
        self.assertAlmostEqual(p[0,2],  0.0559466810510877933, delta=1e-12)

    def test_S2pv(self):
        pv = pysofa.S2pv(-3.21, 0.123, 0.456, -7.8e-6, 9.01e-6, -1.23e-5)

        self.assertAlmostEqual(pv[0,0], -0.4514964673880165228, delta=1e-12)
        self.assertAlmostEqual(pv[0,1],  0.0309339427734258688, delta=1e-12)
        self.assertAlmostEqual(pv[0,2],  0.0559466810510877933, delta=1e-12)

        self.assertAlmostEqual(pv[1,0],  0.1292270850663260170e-4, delta=1e-16)
        self.assertAlmostEqual(pv[1,1],  0.2652814182060691422e-5, delta=1e-16)
        self.assertAlmostEqual(pv[1,2],  0.2568431853930292259e-5, delta=1e-16)

    def test_S2xpv(self):
        s1 = 2.0
        s2 = 3.0

        pv = ndarray(shape=(2,3))
        pv[0][0] =  0.3
        pv[0][1] =  1.2
        pv[0][2] = -2.5

        pv[1][0] =  0.5
        pv[1][1] =  2.3
        pv[1][2] = -0.4

        spv = pysofa.S2xpv(s1, s2, pv)

        self.assertAlmostEqual(spv[0,0],  0.6, delta=1e-12)
        self.assertAlmostEqual(spv[0,1],  2.4, delta=1e-12)
        self.assertAlmostEqual(spv[0,2], -5.0, delta=1e-12)

        self.assertAlmostEqual(spv[1,0],  1.5, delta=1e-12)
        self.assertAlmostEqual(spv[1,1],  6.9, delta=1e-12)
        self.assertAlmostEqual(spv[1,2], -1.2, delta=1e-12)

    def test_Sepp(self):
        a = (1., 0.1, 0.2)
        b = (-3.0, 1e-3, 0.2)

        s = pysofa.Sepp(a, b)

        self.assertAlmostEqual(s, 2.860391919024660768, delta=1e-12)

    def test_Seps(self):
        al =  1.0
        ap =  0.1

        bl =  0.2
        bp = -3.0

        s = pysofa.Seps(al, ap, bl, bp)

        self.assertAlmostEqual(s, 2.346722016996998842, delta=1e-14)

    def test_Sp00(self):
        self.assertAlmostEqual(pysofa.Sp00(2400000.5, 52541.0),
                                    -0.6216698469981019309e-11, delta=1e-12)

    def test_Starpm(self):
        ra1 =   0.01686756
        dec1 = -1.093989828
        pmr1 = -1.78323516e-5
        pmd1 =  2.336024047e-6
        px1 =   0.74723
        rv1 = -21.6

        ra2, dec2, pmr2, pmd2, px2, rv2, j = pysofa.Starpm(ra1, dec1, pmr1,
                                                            pmd1, px1, rv1,
                                                            2400000.5, 50083.0,
                                                            2400000.5, 53736.0)

        self.assertAlmostEqual(ra2, 0.01668919069414242368, delta=1e-13)
        self.assertAlmostEqual(dec2, -1.093966454217127879, delta=1e-13)
        self.assertAlmostEqual(pmr2, -0.1783662682155932702e-4, delta=1e-17)
        self.assertAlmostEqual(pmd2, 0.2338092915987603664e-5, delta=1e-17)
        self.assertAlmostEqual(px2, 0.7473533835323493644, delta=1e-13)
        self.assertAlmostEqual(rv2, -21.59905170476860786, delta=1e-11)

        self.assertEqual(j, 0)

    def test_Starpv(self):
        ra =   0.01686756
        dec = -1.093989828
        pmr = -1.78323516e-5
        pmd =  2.336024047e-6
        px =   0.74723
        rv = -21.6

        pv, j = pysofa.Starpv(ra, dec, pmr, pmd, px, rv)

        self.assertAlmostEqual(pv[0,0], 126668.5912743160601, delta=1e-10)
        self.assertAlmostEqual(pv[0,1], 2136.792716839935195, delta=1e-12)
        self.assertAlmostEqual(pv[0,2], -245251.2339876830091, delta=1e-10)

        self.assertAlmostEqual(pv[1,0], -0.4051854035740712739e-2, delta=1e-13)
        self.assertAlmostEqual(pv[1,1], -0.6253919754866173866e-2, delta=1e-15)
        self.assertAlmostEqual(pv[1,2], 0.1189353719774107189e-1, delta=1e-13)

        self.assertEqual(j, 0)

    def test_Sxp(self):
        s = 2.0

        p = (0.3, 1.2, -2.5)

        sp = pysofa.Sxp(s, p)

        self.assertAlmostEqual(sp[0,0],  0.6, delta=0.0)
        self.assertAlmostEqual(sp[0,1],  2.4, delta=0.0)
        self.assertAlmostEqual(sp[0,2], -5.0, delta=0.0)

    def test_Sxpv(self):
        s = 2.0

        pv = ndarray(shape=(2,3))
        pv[0][0] =  0.3
        pv[0][1] =  1.2
        pv[0][2] = -2.5

        pv[1][0] =  0.5
        pv[1][1] =  3.2
        pv[1][2] = -0.7

        spv = pysofa.Sxpv(s, pv)

        self.assertAlmostEqual(spv[0,0],  0.6, delta=0.0)
        self.assertAlmostEqual(spv[0,1],  2.4, delta=0.0)
        self.assertAlmostEqual(spv[0,2], -5.0, delta=0.0)

        self.assertAlmostEqual(spv[1,0],  1.0, delta=0.0)
        self.assertAlmostEqual(spv[1,1],  6.4, delta=0.0)
        self.assertAlmostEqual(spv[1,2], -1.4, delta=0.0)

    def test_Taitt(self):
        t1, t2, s = pysofa.Taitt(2453750.5, 0.892482639)

        self.assertAlmostEqual(t1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(t2, 0.892855139, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Taiut1(self):
        u1, u2, s = pysofa.Taiut1(2453750.5, 0.892482639, -32.6659)

        self.assertAlmostEqual(u1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(u2, 0.8921045614537037037, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Taiutc(self):
        u1, u2, s = pysofa.Taiutc(2453750.5, 0.892482639)

        self.assertAlmostEqual(u1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(u2, 0.8921006945555555556, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Tcbtdb(self):
        b1, b2, s = pysofa.Tcbtdb(2453750.5, 0.893019599)

        self.assertAlmostEqual(b1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(b2, 0.8928551362746343397, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Tcgtt(self):
        t1, t2, s = pysofa.Tcgtt(2453750.5, 0.892862531)

        self.assertAlmostEqual(t1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(t2, 0.8928551387488816828, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Tdbtcb(self):
        b1, b2, s = pysofa.Tdbtcb(2453750.5, 0.892855137)

        self.assertAlmostEqual(b1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(b2, 0.8930195997253656716, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Tdbtt(self):
        t1, t2, s = pysofa.Tdbtt(2453750.5, 0.892855137, -0.000201)

        self.assertAlmostEqual(t1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(t2, 0.8928551393263888889, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Tf2a(self):
        a, s = pysofa.Tf2a('+', 4, 58, 20.2)

        self.assertAlmostEqual(a, 1.301739278189537429, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Tf2d(self):
        d, s = pysofa.Tf2d(' ', 23, 55, 10.9)

        self.assertAlmostEqual(d, 0.9966539351851851852, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Tr(self):
        r = ndarray(shape=(3,3))
        r[0][0] = 2.0
        r[0][1] = 3.0
        r[0][2] = 2.0

        r[1][0] = 3.0
        r[1][1] = 2.0
        r[1][2] = 3.0

        r[2][0] = 3.0
        r[2][1] = 4.0
        r[2][2] = 5.0

        rt = pysofa.Tr(r)

        self.assertAlmostEqual(rt[0,0], 2.0, delta=0.0)
        self.assertAlmostEqual(rt[0,1], 3.0, delta=0.0)
        self.assertAlmostEqual(rt[0,2], 3.0, delta=0.0)

        self.assertAlmostEqual(rt[1,0], 3.0, delta=0.0)
        self.assertAlmostEqual(rt[1,1], 2.0, delta=0.0)
        self.assertAlmostEqual(rt[1,2], 4.0, delta=0.0)

        self.assertAlmostEqual(rt[2,0], 2.0, delta=0.0)
        self.assertAlmostEqual(rt[2,1], 3.0, delta=0.0)
        self.assertAlmostEqual(rt[2,2], 5.0, delta=0.0)

    def test_Trxp(self):
        r = ndarray(shape=(3,3))
        r[0][0] = 2.0
        r[0][1] = 3.0
        r[0][2] = 2.0

        r[1][0] = 3.0
        r[1][1] = 2.0
        r[1][2] = 3.0

        r[2][0] = 3.0
        r[2][1] = 4.0
        r[2][2] = 5.0

        p = (0.2, 1.5, 0.1)

        trp = pysofa.Trxp(r, p)

        self.assertAlmostEqual(trp[0,0], 5.2, delta=1e-12)
        self.assertAlmostEqual(trp[0,1], 4.0, delta=1e-12)
        self.assertAlmostEqual(trp[0,2], 5.4, delta=1e-12)

    def test_Trxpv(self):
        r = ndarray(shape=(3,3))
        r[0][0] = 2.0
        r[0][1] = 3.0
        r[0][2] = 2.0

        r[1][0] = 3.0
        r[1][1] = 2.0
        r[1][2] = 3.0

        r[2][0] = 3.0
        r[2][1] = 4.0
        r[2][2] = 5.0

        pv = ndarray(shape=(2,3))
        pv[0][0] = 0.2
        pv[0][1] = 1.5
        pv[0][2] = 0.1

        pv[1][0] = 1.5
        pv[1][1] = 0.2
        pv[1][2] = 0.1

        trpv = pysofa.Trxpv(r, pv)

        self.assertAlmostEqual(trpv[0,0], 5.2, delta=1e-12)
        self.assertAlmostEqual(trpv[0,1], 4.0, delta=1e-12)
        self.assertAlmostEqual(trpv[0,2], 5.4, delta=1e-12)

        self.assertAlmostEqual(trpv[1,0], 3.9, delta=1e-12)
        self.assertAlmostEqual(trpv[1,1], 5.3, delta=1e-12)
        self.assertAlmostEqual(trpv[1,2], 4.1, delta=1e-12)

    def test_Tttai(self):
        t1, t2, s = pysofa.Tttai(2453750.5, 0.892482639)

        self.assertAlmostEqual(t1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(t2, 0.892110139, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Tttcg(self):
        t1, t2, s = pysofa.Tttcg(2453750.5, 0.892482639)

        self.assertAlmostEqual(t1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(t2, 0.8924900312508587113, delta=1e-12)
        self.assertEqual(s, 0)

    def test_Tttdb(self):
        b1, b2, j = pysofa.Tttdb(2453750.5, 0.892855139, -0.000201)

        self.assertAlmostEqual(b1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(b2, 0.8928551366736111111, delta=1e-12)
        self.assertEqual(j, 0)

    def test_Ttut1(self):
        u1, u2, j = pysofa.Ttut1(2453750.5, 0.892855139, 64.8499)

        self.assertAlmostEqual(u1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(u2, 0.8921045614537037037, delta=1e-12)
        self.assertEqual(j, 0)

    def test_Ut1tai(self):
        a1, a2, j = pysofa.Ut1tai(2453750.5, 0.892104561, -32.6659)

        self.assertAlmostEqual(a1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(a2, 0.8924826385462962963, delta=1e-12)
        self.assertEqual(j, 0)

    def test_Ut1tt(self):
        t1, t2, j = pysofa.Ut1tt(2453750.5, 0.892104561, 64.8499)

        self.assertAlmostEqual(t1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(t2, 0.8928551385462962963, delta=1e-12)
        self.assertEqual(j, 0)

    def test_Ut1utc(self):
        u1, u2, j = pysofa.Ut1utc(2453750.5, 0.892104561, 0.3341)

        self.assertAlmostEqual(u1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(u2, 0.8921006941018518519, delta=1e-12)
        self.assertEqual(j, 0)

    def test_Utctai(self):
        u1, u2, j = pysofa.Utctai(2453750.5, 0.892100694)

        self.assertAlmostEqual(u1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(u2, 0.8924826384444444444, delta=1e-12)
        self.assertEqual(j, 0)

    def test_Utcut1(self):
        u1, u2, j = pysofa.Utcut1(2453750.5, 0.892100694, 0.3341)

        self.assertAlmostEqual(u1, 2453750.5, delta=1e-6)
        self.assertAlmostEqual(u2, 0.8921045608981481481, delta=1e-12)
        self.assertEqual(j, 0)

    def test_Xy06(self):
        x, y = pysofa.Xy06(2400000.5, 53736.0)

        self.assertAlmostEqual(x, 0.5791308486706010975e-3, delta=1e-15)
        self.assertAlmostEqual(y, 0.4020579816732958141e-4, delta=1e-16)

    def test_Xys00a(self):
        x, y, s = pysofa.Xys00a(2400000.5, 53736.0)

        self.assertAlmostEqual(x,  0.5791308472168152904e-3, delta=1e-14)
        self.assertAlmostEqual(y,  0.4020595661591500259e-4, delta=1e-15)
        self.assertAlmostEqual(s, -0.1220040848471549623e-7, delta=1e-18)

    def test_Xys00b(self):
        x, y, s = pysofa.Xys00b(2400000.5, 53736.0)

        self.assertAlmostEqual(x,  0.5791301929950208873e-3, delta=1e-14)
        self.assertAlmostEqual(y,  0.4020553681373720832e-4, delta=1e-15)
        self.assertAlmostEqual(s, -0.1220027377285083189e-7, delta=1e-18)

    def test_Xys06a(self):
        x, y, s = pysofa.Xys06a(2400000.5, 53736.0)

        self.assertAlmostEqual(x,  0.5791308482835292617e-3, delta=1e-14)
        self.assertAlmostEqual(y,  0.4020580099454020310e-4, delta=1e-15)
        self.assertAlmostEqual(s, -0.1220032294164579896e-7, delta=1e-18)

    def test_Zp(self):
        p = ndarray(shape=(3))
        p[0] =  0.3
        p[1] =  1.2
        p[2] = -2.5

        pysofa.Zp(p)

        self.assertAlmostEqual(p[0], 0.0, delta=0.0)
        self.assertAlmostEqual(p[1], 0.0, delta=0.0)
        self.assertAlmostEqual(p[2], 0.0, delta=0.0)

    def test_Zpv(self):
        pv = ndarray(shape=(2,3))
        pv[0][0] =  0.3
        pv[0][1] =  1.2
        pv[0][2] = -2.5

        pv[1][0] = -0.5
        pv[1][1] =  3.1
        pv[1][2] =  0.9

        pysofa.Zpv(pv)

        self.assertAlmostEqual(pv[0][0], 0.0, delta=0.0)
        self.assertAlmostEqual(pv[0][1], 0.0, delta=0.0)
        self.assertAlmostEqual(pv[0][2], 0.0, delta=0.0)

        self.assertAlmostEqual(pv[1][0], 0.0, delta=0.0)
        self.assertAlmostEqual(pv[1][1], 0.0, delta=0.0)
        self.assertAlmostEqual(pv[1][2], 0.0, delta=0.0)

    def test_Zr(self):
        r = ndarray(shape=(3,3), dtype=float)
        r[0][0] = 2.0
        r[1][0] = 3.0
        r[2][0] = 2.0

        r[0][1] = 3.0
        r[1][1] = 2.0
        r[2][1] = 3.0

        r[0][2] = 3.0
        r[1][2] = 4.0
        r[2][2] = 5.0

        pysofa.Zr(r)

        self.assertAlmostEqual(r[0][0], 0.0, delta=0.0)
        self.assertAlmostEqual(r[1][0], 0.0, delta=0.0)
        self.assertAlmostEqual(r[2][0], 0.0, delta=0.0)

        self.assertAlmostEqual(r[0][1], 0.0, delta=0.0)
        self.assertAlmostEqual(r[1][1], 0.0, delta=0.0)
        self.assertAlmostEqual(r[2][1], 0.0, delta=0.0)

        self.assertAlmostEqual(r[0][2], 0.0, delta=0.0)
        self.assertAlmostEqual(r[1][2], 0.0, delta=0.0)
        self.assertAlmostEqual(r[2][2], 0.0, delta=0.0)




if __name__ == '__main__':
    unittest.main()

