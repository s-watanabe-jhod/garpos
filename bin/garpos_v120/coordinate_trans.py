"""
Created:
    07/01/2020 by S. Watanabe
Contents:
    corr_attitude(rx, ry, rz, thy, thr, thp)
    llh2xyz(lt, ln, hgt)
    xyz2enu(x, y, z, lat0, lon0, hgt0, inv=1)
"""
import sys
import math
import numpy as np

def corr_attitude(rx, ry, rz, thy, thr, thp):
    """
    Calculate transducer's position from GNSS antenna in ENU coordinate.

    Supports both scalar and array inputs for thy/thr/thp.

    Parameters
    ----------
    rx : float
        Forward position of transducer from GNSS ant. (in vessel's coord.)
    ry : float
        Rightward position of transducer from GNSS ant. (in vessel's coord.)
    rz : float
        Downward position of transducer from GNSS ant. (in vessel's coord.)
    thy : float or ndarray
        Yaw/Heading in degree
    thr : float or ndarray
        Roll in degree
    thp : float or ndarray
        Pitch in degree

    Returns
    -------
    pole_de : float or ndarray
        Eastward transducer's position from GNSS ant.
    pole_dn : float or ndarray
        Northward transducer's position from GNSS ant.
    pole_du : float or ndarray
        Upward transducer's position from GNSS ant.
    """

    yw = np.asarray(thy) * (np.pi/180.)
    rl = np.asarray(thr) * (np.pi/180.)
    pc = np.asarray(thp) * (np.pi/180.)

    crl = np.cos(rl)
    srl = np.sin(rl)
    cpc = np.cos(pc)
    spc = np.sin(pc)
    cyw = np.cos(yw)
    syw = np.sin(yw)

    # trans = tr_yw @ tr_pc @ tr_rl, applied to [rx, ry, rz]
    # Expand matrix multiply element-wise for vectorized angles:
    # row0 of (tr_yw @ tr_pc): [cyw*cpc, -syw, cyw*spc]
    # row1 of (tr_yw @ tr_pc): [syw*cpc,  cyw, syw*spc]
    # row2 of (tr_yw @ tr_pc): [-spc,     0,   cpc    ]
    # Then multiply by tr_rl on the right and dot with [rx, ry, rz]:
    # dned[0] = (cyw*cpc)*rx + (cyw*spc*srl - syw*crl)*ry + (cyw*spc*crl + syw*srl)*rz
    # dned[1] = (syw*cpc)*rx + (syw*spc*srl + cyw*crl)*ry + (syw*spc*crl - cyw*srl)*rz
    # dned[2] = (-spc)*rx    + (cpc*srl)*ry               + (cpc*crl)*rz

    dned0 = (cyw*cpc)*rx + (cyw*spc*srl - syw*crl)*ry + (cyw*spc*crl + syw*srl)*rz
    dned1 = (syw*cpc)*rx + (syw*spc*srl + cyw*crl)*ry + (syw*spc*crl - cyw*srl)*rz
    dned2 = (-spc)*rx    + (cpc*srl)*ry               + (cpc*crl)*rz

    pole_de =  dned1
    pole_dn =  dned0
    pole_du = -dned2

    return pole_de, pole_dn, pole_du


def llh2xyz(lt, ln, hgt):
    """
    Convert lat, long, height in WGS84 to ECEF (X,Y,Z).
    lat and long given in decimal degrees.
    height should be given in meters

    Parameters
    ----------
    lt : float
        Latitude in degrees
    ln : float
        Longitude in degrees
    hgt : float
        Height in meters

    Returns
    -------
    X : float
        X (m) in ECEF
    Y : float
        Y (m) in ECEF
    Z : float
        Z (m) in ECEF
    """
    lat = lt * math.pi/180.
    lon = ln * math.pi/180.
    a  = 6378137.0          # earth semimajor axis in meters
    f  = 1.0/298.257223563  # reciprocal flattening
    e2 = 2.0 * f - f**2     # eccentricity squared

    chi = ( 1.0 - e2*(math.sin(lat))**2)**0.5
    b = a*(1.-e2)

    X = (a/chi + hgt) * math.cos(lat) * math.cos(lon)
    Y = (a/chi + hgt) * math.cos(lat) * math.sin(lon)
    Z = (b/chi + hgt) * math.sin(lat)

    return X, Y, Z


def xyz2enu(x, y, z, lat0, lon0, hgt0, inv=1):
    """
    Rotates the vector of positions XYZ and covariance to
    the local east-north-up system at latitude and longitude
    (or XYZ coordinates) specified in origin.
    if inv = -1. then enu -> xyz

    Parameters
    ----------
    x : float
    y : float
    z : float
        Position in ECEF (if inv=-1, in ENU)
    lat0 : float
    lon0 : float
    Hgt0 : float
        Origin for the local system in degrees.
    inv : 1 or -1
        Switch (1: XYZ -> ENU, -1: ENU -> XYZ)

    Returns
    -------
    e : float
    n : float
    u : float
        Position in ENU (if inv=-1, in ECEF)
    """

    if inv != 1 and inv != -1:
        print("error in xyz2enu : ", inv)
        sys.exit(1)

    lat  = lat0 * math.pi/180. * inv
    lon  = lon0 * math.pi/180. * inv

    sphi = math.sin(lat)
    cphi = math.cos(lat)
    slmb = math.sin(lon)
    clmb = math.cos(lon)

    T1 = [     -slmb,       clmb,    0]
    T2 = [-sphi*clmb, -sphi*slmb, cphi]
    T3 = [ cphi*clmb,  cphi*slmb, sphi]

    e = x * T1[0] + y * T1[1] + z * T1[2]
    n = x * T2[0] + y * T2[1] + z * T2[2]
    u = x * T3[0] + y * T3[1] + z * T3[2]

    return e, n, u
    
