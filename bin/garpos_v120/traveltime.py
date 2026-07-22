"""
Created:
    07/01/2020 by S. Watanabe
Modified:
    02/13/2025 to add azimuth (by S. Watanabe)
    07/18/2026 cache library handle; add batch raytrace
"""
import sys
import math
import ctypes
import numpy as np

_f90_cache = {}


def _get_f90(icfg):
    """Load and cache the Fortran shared library handle."""
    libdir = icfg.get("Inv-parameter", "lib_directory")
    lib_raytrace = icfg.get("Inv-parameter", "lib_raytrace")
    key = (libdir, lib_raytrace)
    if key not in _f90_cache:
        f90 = np.ctypeslib.load_library(lib_raytrace, libdir)
        f90.raytrace_.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
        ]
        f90.raytrace_.restype = ctypes.c_void_p
        _f90_cache[key] = f90
    return _f90_cache[key]


def _call_raytrace(f90, dst, yd, ys, l_depth, l_speed):
    """Call Fortran raytrace_ on pre-validated arrays."""
    n = len(dst)
    nn = ctypes.byref(ctypes.c_int32(n))
    nl = ctypes.byref(ctypes.c_int32(len(l_depth)))
    dsv = np.zeros(n)
    ctm = np.zeros(n)
    cag = np.zeros(n)
    f90.raytrace_(nn, nl, l_depth, l_speed, dst, yd, ys, dsv, ctm, cag)
    return ctm, cag


def calc_traveltime(shotdat, mp, nMT, icfg, svp):
    """
    Calculate the round-trip travel time.

    Parameters
    ----------
    shotdat : DataFrame
        GNSS-A shot dataset.
    mp : ndarray
        complete model parameter vector.
    nMT : int
        number of transponders.
    icfg : configparser
        Config file for inversion conditions.
    svp : DataFrame
        Sound speed profile.

    Returns
    -------
    calTT : ndarray
        Calculated travel time (sec.).
    calA0 : ndarray
        Calculated take-off angle (degree).
    azimuth : ndarray
        Azimuth angle (degree).
    """

    f90 = _get_f90(icfg)

    # station pos
    sta0_e = mp[shotdat['mtid']+0] + mp[nMT*3+0]
    sta0_n = mp[shotdat['mtid']+1] + mp[nMT*3+1]
    sta0_u = mp[shotdat['mtid']+2] + mp[nMT*3+2]

    e0 = shotdat.ant_e0.values + shotdat.ple0.values
    n0 = shotdat.ant_n0.values + shotdat.pln0.values
    u0 = shotdat.ant_u0.values + shotdat.plu0.values
    e1 = shotdat.ant_e1.values + shotdat.ple1.values
    n1 = shotdat.ant_n1.values + shotdat.pln1.values
    u1 = shotdat.ant_u1.values + shotdat.plu1.values

    dist0 = ((e0 - sta0_e)**2. + (n0 - sta0_n)**2.)**0.5
    dist1 = ((e1 - sta0_e)**2. + (n1 - sta0_n)**2.)**0.5
    azimuth0 = np.arctan2(e0 - sta0_e, n0 - sta0_n) * 180./np.pi
    azimuth1 = np.arctan2(e1 - sta0_e, n1 - sta0_n) * 180./np.pi
    azimuth = (azimuth0 + azimuth1) / 2.

    dst = np.append(dist0, dist1)
    yd  = np.append(sta0_u, sta0_u)
    ys  = np.append(u0, u1)

    # sv layer
    l_depth = svp.depth.values
    l_speed = svp.speed.values

    if np.isnan(yd).any():
        print(yd[np.isnan(yd)])
        print("nan in yd")
        sys.exit(1)
    if np.isnan(ys).any():
        print(ys[np.isnan(ys)])
        print("nan in ys")
        sys.exit(1)

    if min(yd) < -l_depth[-1]:
        print(min(yd) , -l_depth[-1])
        print("yd is deeper than layer")
        print(mp[0:15] , mp[nMT*3+2])
        sys.exit(1)
    if max(ys) > -l_depth[0]:
        l_depth = np.append(-40.,l_depth)
        l_speed = np.append(l_speed[0],l_speed)
        if len(ys[ys > -l_depth[0]]) > 50:
            print(ys[ys > -l_depth[0]] , -l_depth[0])
            print("many of ys are shallower than layer")
            print(l_depth)
            sys.exit(1)
        if max(ys) > -l_depth[0]:
            print(max(ys) , -l_depth[0])
            print(ys[ys > -l_depth[0]] , -l_depth[0])
            print("ys is shallower than layer")
            print(l_depth)
            sys.exit(1)

    ndat = len(shotdat.index)
    ctm, cag = _call_raytrace(f90, dst, yd, ys, l_depth, l_speed)

    calA0 = 180. - (cag[:ndat] + cag[ndat:])/2. * 180./math.pi
    calTT = ctm[:ndat] + ctm[ndat:]

    return calTT, calA0, azimuth


def calc_traveltime_batch(shotdat_list, mp_list, nMT, icfg, svp):
    """
    Batch travel time calculation for multiple perturbations.

    Each (shotdat, mp) pair produces one set of TT/angle results.
    All pairs are concatenated into a single Fortran call.

    Parameters
    ----------
    shotdat_list : list of DataFrame
    mp_list : list of ndarray
    nMT : int
    icfg : configparser
    svp : DataFrame

    Returns
    -------
    results : list of (calTT, calA0, azimuth) tuples
    """
    f90 = _get_f90(icfg)

    l_depth = svp.depth.values.copy()
    l_speed = svp.speed.values.copy()

    all_dst = []
    all_yd = []
    all_ys = []
    ndats = []
    azimuths = []
    needs_svp_extend = False

    for shotdat, mp in zip(shotdat_list, mp_list):
        sta0_e = mp[shotdat['mtid']+0] + mp[nMT*3+0]
        sta0_n = mp[shotdat['mtid']+1] + mp[nMT*3+1]
        sta0_u = mp[shotdat['mtid']+2] + mp[nMT*3+2]

        e0 = shotdat.ant_e0.values + shotdat.ple0.values
        n0 = shotdat.ant_n0.values + shotdat.pln0.values
        u0 = shotdat.ant_u0.values + shotdat.plu0.values
        e1 = shotdat.ant_e1.values + shotdat.ple1.values
        n1 = shotdat.ant_n1.values + shotdat.pln1.values
        u1 = shotdat.ant_u1.values + shotdat.plu1.values

        dist0 = ((e0 - sta0_e)**2. + (n0 - sta0_n)**2.)**0.5
        dist1 = ((e1 - sta0_e)**2. + (n1 - sta0_n)**2.)**0.5
        azimuth0 = np.arctan2(e0 - sta0_e, n0 - sta0_n) * 180./np.pi
        azimuth1 = np.arctan2(e1 - sta0_e, n1 - sta0_n) * 180./np.pi
        azimuths.append((azimuth0 + azimuth1) / 2.)

        dst = np.append(dist0, dist1)
        yd  = np.append(sta0_u, sta0_u)
        ys  = np.append(u0, u1)

        if max(ys) > -l_depth[0]:
            needs_svp_extend = True

        all_dst.append(dst)
        all_yd.append(yd)
        all_ys.append(ys)
        ndats.append(len(shotdat.index))

    if needs_svp_extend:
        l_depth = np.append(-40., l_depth)
        l_speed = np.append(l_speed[0], l_speed)

    big_dst = np.concatenate(all_dst)
    big_yd = np.concatenate(all_yd)
    big_ys = np.concatenate(all_ys)

    ctm, cag = _call_raytrace(f90, big_dst, big_yd, big_ys, l_depth, l_speed)

    results = []
    offset = 0
    for i, ndat in enumerate(ndats):
        seg = ndat * 2
        ct = ctm[offset:offset+seg]
        ca = cag[offset:offset+seg]
        calTT = ct[:ndat] + ct[ndat:]
        calA0 = 180. - (ca[:ndat] + ca[ndat:])/2. * 180./math.pi
        results.append((calTT, calA0, azimuths[i]))
        offset += seg

    return results
