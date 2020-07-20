"""
Created:
	07/01/2020 by S. Watanabe
"""
import sys
import math
import ctypes
import numpy as np

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
	"""
	
	# fortran library
	libdir = icfg.get("Inv-parameter","lib_directory")
	lib_raytrace = icfg.get("Inv-parameter","lib_raytrace")
	
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
	
	dst = np.append(dist0, dist1)
	yd  = np.append(sta0_u, sta0_u)
	ys  = np.append(u0, u1)
	dsv = np.zeros(len(dst))
	
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
	
	#######################################
	# for call f90 library (calc 1way TT) #
	#######################################
	ndat = len(shotdat.index)
	nl = len(l_depth)
	nn = ctypes.byref(ctypes.c_int32(ndat*2))
	nl = ctypes.byref(ctypes.c_int32(nl))
	
	# output
	ctm = np.zeros_like(dst)
	cag = np.zeros_like(dst)
	f90 = np.ctypeslib.load_library(lib_raytrace, libdir)
	f90.raytrace_.argtypes = [
		ctypes.POINTER(ctypes.c_int32), # n
		ctypes.POINTER(ctypes.c_int32), # nlyr
		np.ctypeslib.ndpointer(dtype=np.float64), # l_depth
		np.ctypeslib.ndpointer(dtype=np.float64), # l_speed
		np.ctypeslib.ndpointer(dtype=np.float64), # dist
		np.ctypeslib.ndpointer(dtype=np.float64), # yd
		np.ctypeslib.ndpointer(dtype=np.float64), # ys
		np.ctypeslib.ndpointer(dtype=np.float64), # dsv
		np.ctypeslib.ndpointer(dtype=np.float64), # ctm (output)
		np.ctypeslib.ndpointer(dtype=np.float64)  # cag (output)
		]
	f90.raytrace_.restype = ctypes.c_void_p
	
	f90.raytrace_(nn, nl, l_depth, l_speed, dst, yd, ys, dsv, ctm, cag)
	
	calTime  = np.array(ctm)
	calAngle = np.array(cag)
	
	#############################
	
	calA0 = 180. - (calAngle[:ndat] + calAngle[ndat:])/2. * 180./math.pi
	calTT = calTime[:ndat] + calTime[ndat:]
	
	return calTT, calA0
	
