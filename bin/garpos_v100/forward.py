"""
Created:
	07/01/2020 by S. Watanabe
Contains:
	calc_forward
	calc_gamma
	jacobian_pos
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import lil_matrix

# garpos module
from .coordinate_trans import corr_attitude
from .traveltime import calc_traveltime


def calc_forward(shots, mp, nMT, icfg, svp, T0):
	"""
	Calculate the forward modeling of observation eqs.

	Parameters
	----------
	shots : DataFrame
		GNSS-A shot dataset.
	mp : ndarray
		complete model parameter vector.
	nMT : int
		number of transponders.
	icfg : configparser
		Config file for inversion conditions.
	svp : DataFrame
		Sound speed profile.
	T0 : float
		Typical travel time.

	Returns
	-------
	shots : DataFrame
		GNSS-A shot dataset in which calculated data is added.
	"""

	rsig = float(icfg.get("Inv-parameter","RejectCriteria"))

	# calc ATD offset
	calATD = np.vectorize(corr_attitude)
	pl0 = mp[(nMT+1)*3+0]
	pl1 = mp[(nMT+1)*3+1]
	pl2 = mp[(nMT+1)*3+2]
	hd0 = shots.head0.values
	hd1 = shots.head1.values
	rl0 = shots.roll0.values
	rl1 = shots.roll1.values
	pc0 = shots.pitch0.values
	pc1 = shots.pitch1.values
	ple0, pln0, plu0 = calATD(pl0, pl1, pl2, hd0, rl0, pc0)
	ple1, pln1, plu1 = calATD(pl0, pl1, pl2, hd1, rl1, pc1)
	shots['ple0'] = ple0
	shots['pln0'] = pln0
	shots['plu0'] = plu0
	shots['ple1'] = ple1
	shots['pln1'] = pln1
	shots['plu1'] = plu1

	# calc Residuals
	cTT, cTO = calc_traveltime(shots, mp, nMT, icfg, svp)
	logTTc = np.log( cTT/T0 ) - shots.gamma.values
	ResiTT = shots.logTT.values - logTTc

	shots['calcTT'] = cTT
	shots['TakeOff'] = cTO
	shots['logTTc'] = logTTc
	shots['ResiTT'] = ResiTT
	# approximation log(1 + x) ~ x
	shots['ResiTTreal'] = ResiTT * shots.TT.values

	if rsig > 0.1:
		aveRTT = shots[~shots['flag']].ResiTT.mean()
		sigRTT = shots[~shots['flag']].ResiTT.std()
		th0 = aveRTT + rsig * sigRTT
		th1 = aveRTT - rsig * sigRTT
		shots['flag'] = (shots['ResiTT'] > th0) | (shots['ResiTT'] < th1) | shots['iniflag']
		aveRTT1 = shots[~shots['flag']].ResiTT.mean()
		sigRTT1 = shots[~shots['flag']].ResiTT.std()

	return shots


def calc_gamma(mp, shotdat, imp0, spdeg, knots):
	"""
	Calculate correction value "gamma" in the observation eqs.

	Parameters
	----------
	mp : ndarray
		complete model parameter vector.
	shotdat : DataFrame
		GNSS-A shot dataset.
	imp0 : ndarray (len=5)
		Indices where the type of model parameters change.
	p : int
		spline degree (=3).
	knots : list of ndarray (len=5)
		B-spline knots for each component in "gamma".

	Returns
	-------
	gamma : ndarray
		Values of "gamma". Note that scale facter is not applied.
	a : 2-d list of ndarray
		[a0[<alpha>], a1[<alpha>]] :: a[<alpha>] at transmit/received time.
		<alpha> is corresponding to <0>, <1E>, <1N>, <2E>, <2N>.
	"""

	a0 = []
	a1 = []
	for k, kn in enumerate(knots):
		if len(kn) == 0:
			a0.append( 0. )
			a1.append( 0. )
			continue
		ct = mp[imp0[k]:imp0[k+1]]
		bs = BSpline(kn, ct, spdeg, extrapolate=False)
		a0.append( bs(shotdat.ST.values) )
		a1.append( bs(shotdat.RT.values) )

	ls = 1000.  # m/s/m to m/s/km order for gradient

	de0 = shotdat.de0.values
	de1 = shotdat.de1.values
	dn0 = shotdat.dn0.values
	dn1 = shotdat.dn1.values
	mte = shotdat.mtde.values
	mtn = shotdat.mtdn.values

	gamma0_0 =  a0[0]
	gamma0_1 = (a0[1] * de0 + a0[2] * dn0) / ls
	gamma0_2 = (a0[3] * mte + a0[4] * mtn) / ls

	gamma1_0 =  a1[0]
	gamma1_1 = (a1[1] * de1 + a1[2] * dn1) / ls
	gamma1_2 = (a1[3] * mte + a1[4] * mtn) / ls

	gamma0 = gamma0_0 + gamma0_1 + gamma0_2
	gamma1 = gamma1_0 + gamma1_1 + gamma1_2

	gamma = (gamma0 + gamma1)/2.
	a = [a0, a1]

	return gamma, a


def jacobian_pos(icfg, mp, slvidx0, shotdat, mtidx, svp, T0):
	"""
	Calculate Jacobian matrix for positions.

	Parameters
	----------
	icfg : configparser
		Config file for inversion conditions.
	mp : ndarray
		complete model parameter vector.
	slvidx0 : list
		Indices of model parameters to be solved.
	shotdat : DataFrame
		GNSS-A shot dataset.
	mtidx : dictionary
		Indices of mp for each MT.
	svp : DataFrame
		Sound speed profile.
	T0 : float
		Typical travel time.

	Returns
	-------
	jcbpos : lil_matrix
		Jacobian matrix for positions.
	"""

	# read inversion parameters
	deltap = float(icfg.get("Inv-parameter","deltap"))
	deltab = float(icfg.get("Inv-parameter","deltab"))

	ndata = shotdat.index.size

	MTs = mtidx.keys()
	nMT = len(MTs)
	nmppos = len(slvidx0)

	jcbpos  = lil_matrix( (nmppos, ndata) )
	imp = 0

	gamma = shotdat.gamma.values
	logTTc = shotdat.logTTc.values
	##################################
	### Calc Jacobian for Position ###
	##################################
	# for eastward
	mpj = mp.copy()
	mpj[nMT*3 + 0] += deltap
	cTTj, cTOj = calc_traveltime(shotdat, mpj, nMT, icfg, svp)
	logTTcj = np.log( cTTj/T0 ) - gamma
	shotdat['jacob0'] = (logTTcj - logTTc) / deltap
	# for northward
	mpj = mp.copy()
	mpj[nMT*3 + 1] += deltap
	cTTj, cTOj = calc_traveltime(shotdat, mpj, nMT, icfg, svp)
	logTTcj = np.log( cTTj/T0 ) - gamma
	shotdat['jacob1'] = (logTTcj - logTTc) / deltap
	# for upward
	mpj = mp.copy()
	mpj[nMT*3 + 2] += deltap
	cTTj, cTOj = calc_traveltime(shotdat, mpj, nMT, icfg, svp)
	logTTcj = np.log( cTTj/T0 ) - gamma
	shotdat['jacob2'] = (logTTcj - logTTc) / deltap

	### Jacobian for each MT ###
	for mt in MTs:
		for j in range(3):
			idx = mtidx[mt] + j
			if not (idx in  slvidx0):
				continue
			jccode = "jacob%1d" % j
			shotdat['hit'] = shotdat[jccode] * (shotdat['MT'] == mt)
			jcbpos[imp,:] = np.array([shotdat.hit.values])
			imp += 1

	### Jacobian for Center Pos ###
	for j in range(3):
		idx = nMT*3 + j
		if not (idx in  slvidx0):
			continue
		jccode = "jacob%1d" % j
		jcbpos[imp,:] = shotdat[jccode].values
		imp += 1

	####################################
	### Calc Jacobian for ATD offset ###
	####################################
	for j in range(3): # j = 0:rightward, 1:forward, 2:upward
		idx = nMT*3 + 3 + j
		if not (idx in  slvidx0):
			continue
		# calc Jacobian
		mpj = mp.copy()
		mpj[(nMT+1)*3 + j] += deltap
		tmpj = shotdat.copy()

		# calc ATD offset
		calATD = np.vectorize(corr_attitude)
		pl0 = mpj[(nMT+1)*3 + 0]
		pl1 = mpj[(nMT+1)*3 + 1]
		pl2 = mpj[(nMT+1)*3 + 2]
		hd0 = shotdat.head0.values
		hd1 = shotdat.head1.values
		rl0 = shotdat.roll0.values
		rl1 = shotdat.roll1.values
		pc0 = shotdat.pitch0.values
		pc1 = shotdat.pitch1.values
		ple0, pln0, plu0 = calATD(pl0, pl1, pl2, hd0, rl0, pc0)
		ple1, pln1, plu1 = calATD(pl0, pl1, pl2, hd1, rl1, pc1)
		tmpj['ple0'] = ple0
		tmpj['pln0'] = pln0
		tmpj['plu0'] = plu0
		tmpj['ple1'] = ple1
		tmpj['pln1'] = pln1
		tmpj['plu1'] = plu1

		cTTj, cTOj = calc_traveltime(tmpj, mpj, nMT, icfg, svp)
		logTTcj = np.log( cTTj/T0 ) - gamma
		jcbpos[imp,:] = (logTTcj - logTTc) / deltap
		imp += 1

	return jcbpos
