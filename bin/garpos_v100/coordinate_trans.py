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

	Parameters
	----------
	rx : float
		Forward position of transducer from GNSS ant. (in vessel's coord.)
	ry : float
		Rightward position of transducer from GNSS ant. (in vessel's coord.)
	rz : float
		Downward position of transducer from GNSS ant. (in vessel's coord.)
	thy : float
		Yaw/Heading in degree
	thr : float
		Roll in degree
	thp : float
		Pitch in degree

	Returns
	-------
	pole_de : float
		Eastward transducer's position from GNSS ant.
	pole_dn : float
		Northward transducer's position from GNSS ant.
	pole_du : float
		Upward transducer's position from GNSS ant.
	"""

	yw = thy * math.pi/180.
	rl = thr * math.pi/180.
	pc = thp * math.pi/180.

	crl = math.cos(rl)
	srl = math.sin(rl)
	cpc = math.cos(pc)
	spc = math.sin(pc)
	cyw = math.cos(yw)
	syw = math.sin(yw)

	tr_rl = np.matrix([[ 1.0, 0.0, 0.0],
					   [ 0.0, crl,-srl],
					   [ 0.0, srl, crl]])

	tr_pc = np.matrix([[ cpc, 0.0, spc],
					   [ 0.0, 1.0, 0.0],
					   [-spc, 0.0, cpc]])

	tr_yw = np.matrix([[ cyw,-syw, 0.0],
					   [ syw, cyw, 0.0],
					   [ 0.0, 0.0, 1.0]])

	trans = (tr_yw @ tr_pc) @ tr_rl
	atd  = np.matrix([[rx],[ry],[rz]])
	dned = trans @ atd

	pole_de =  dned[1,0]
	pole_dn =  dned[0,0]
	pole_du = -dned[2,0]

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
	
