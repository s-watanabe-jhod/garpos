"""
Created:
	07/01/2020 by S. Watanabe
Contains:
	init_position
	make_knots
	derivative2
	data_correlation
"""
import sys
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, linalg


def init_position(cfg, denu, MTs):
	"""
	Calculate Jacobian matrix for positions.

	Parameters
	----------
	cfg : configparser
		Config file for site paramters.
	denu : ndarray
		Input of position (mainly for particle filters).
	MTs : list
		List of transponders' name.

	Returns
	-------
	mp : ndarray
		complete model parameter vector. (only for position)
	Dipos : csc_matrix
		A priori covariance for position.
	slvidx0 : list
		Indices of model parameters to be solved. (only for position)
	mtidx : dictionary
		Indices of mp for each MT.
	"""

	mtidx = {}
	mp = np.array([])
	ae = np.array([])
	for imt, mt in enumerate(MTs):
		mtidx[mt] = imt * 3
		dpos = cfg.get("Model-parameter", mt + "_dPos").split()
		dpos = list(map(float, dpos))
		mp = np.append(mp, dpos[0:3])
		ae = np.append(ae, dpos[3:6])

	dcnt = cfg.get("Model-parameter", "dCentPos").split()
	dcnt = list(map(float, dcnt))
	mp = np.append(mp, dcnt[0:3]+denu[0:3])
	ae = np.append(ae, dcnt[3:6])
	if len(dcnt) <= 6:
		covNU = 0.0
		covUE = 0.0
		covEN = 0.0
	else:
		covNU = dcnt[6]
		covUE = dcnt[7]
		covEN = dcnt[8]
	if ae[len(MTs)*3:].sum() > 0.001 and ae[0:len(MTs)*3].sum() > 0.001:
		print("Error: ape for each station must be 0 in rigid-array mode!")
		sys.exit(1)

	atd = cfg.get("Model-parameter", "ATDoffset").split()
	atd = list(map(float, atd))
	mp = np.append(mp, atd[0:3])
	if atd[3] > 1.e-8:
		ae = np.append(ae, 3.0)
	else:
		ae = np.append(ae, 0.0)
	if atd[4] > 1.e-8:
		ae = np.append(ae, 3.0)
	else:
		ae = np.append(ae, 0.0)
	if atd[5] > 1.e-8:
		ae = np.append(ae, 3.0)
	else:
		ae = np.append(ae, 0.0)

	# set a priori variance for position parameters
	D0pos = lil_matrix(np.diag( ae**2. ))
	# set a priori covariance for dCentPos
	D0pos[len(MTs)*3+1,len(MTs)*3+2] = covNU
	D0pos[len(MTs)*3+2,len(MTs)*3+0] = covUE
	D0pos[len(MTs)*3+0,len(MTs)*3+1] = covEN
	D0pos[len(MTs)*3+2,len(MTs)*3+1] = covNU
	D0pos[len(MTs)*3+0,len(MTs)*3+2] = covUE
	D0pos[len(MTs)*3+1,len(MTs)*3+0] = covEN

	slvidx0 = np.where( ae > 1.e-14 )[0]
	nmppos = len(slvidx0)
	Dpos = lil_matrix( (nmppos, nmppos) )
	for i, ipos in enumerate(slvidx0):
		for j, jpos in enumerate(slvidx0):
			Dpos[i, j] = D0pos[ipos,jpos]
	Dpos = Dpos.tocsc()
	Dipos = linalg.inv( Dpos )

	return mp, Dipos, slvidx0, mtidx


def make_knots(shotdat, spdeg, nmpsv):
	"""
	Create the B-spline knots for correction value "gamma".

	Parameters
	----------
	shotdat : DataFrame
		GNSS-A shot dataset.
	spdeg : int
		spline degree (=3).
	nmpsv : list of int (len=5)
		number of knots per subset.

	Returns
	-------
	knots : list of ndarray (len=5)
		B-spline knots for each component in "gamma".
	"""

	sets = shotdat['SET'].unique()
	stf0 = 0.
	setint = []

	st0s = np.array([shotdat.loc[shotdat.SET==s, "ST"].min() for s in sets])
	stfs = np.array([shotdat.loc[shotdat.SET==s, "RT"].max() for s in sets])
	setdurs = stfs - st0s
	setints = st0s[1:] - stfs[:-1]
	setdur = setdurs.mean()

	st0 = shotdat.ST.values.min()
	stf = shotdat.RT.values.max()

	obsdur = stf - st0
	nsetdur = int(obsdur/setdur)
	nknots = [ n * nsetdur for n in nmpsv ]
	knots = [ np.linspace(st0, stf, nall+1) for nall in nknots ]

	for k, cn in enumerate(knots):

		if nknots[k] == 0:
			knots[k] = np.array([])
			continue

		rmknot = np.array([])
		for i in range(len(sets)-1):
			isetkn = np.where( (knots[k]>stfs[i]) & (knots[k]<st0s[i+1]) )[0]
			if len(isetkn) > 2*(spdeg+2):
				rmknot = np.append(rmknot, isetkn[spdeg+1:-spdeg-1])
		knots[k] = np.delete(knots[k], rmknot)

		dkn = (stf-st0)/float(nknots[k])
		addkn0 = np.array( [st0-dkn*(n+1) for n in reversed(range(spdeg))] )
		addknf = np.array( [stf+dkn*(n+1) for n in range(spdeg)] )
		knots[k] = np.append(addkn0, knots[k])
		knots[k] = np.append(knots[k], addknf)

	return knots


def derivative2(imp0, p, knots, lambdas):
	"""
	Calculate the matrix for 2nd derivative of the B-spline basis

	Parameters
	----------
	imp0 : ndarray (len=5)
		Indices where the type of model parameters change.
	p : int
		spline degree (=3).
	knots : list of ndarray (len=5)
		B-spline knots for each component in "gamma".
	lambdas : list of float (len=5)
		Hyperparameter controlling the smoothness of gamma's components.

	Returns
	-------
	H : ndarray
		2nd derivative matrix of the B-spline basis.
	"""
	diff = lil_matrix( (imp0[5], imp0[5]) )

	for k in range(len(lambdas)):

		kn = knots[k]
		if len(kn) == 0:
			continue

		delta =  lil_matrix( (imp0[k+1]-imp0[k]-2, imp0[k+1]-imp0[k]) )
		w = lil_matrix( (imp0[k+1]-imp0[k]-2, imp0[k+1]-imp0[k]-2) )

		for j in range(imp0[k+1]-imp0[k]-2):
			dkn0 = (kn[j+p+1] - kn[j+p  ])/3600.
			dkn1 = (kn[j+p+2] - kn[j+p+1])/3600.

			delta[j,j]   =  1./dkn0
			delta[j,j+1] = -1./dkn0 -1./dkn1
			delta[j,j+2] =  1./dkn1

			if j >= 1:
				w[j,j-1] = dkn0 / 6.
				w[j-1,j] = dkn0 / 6.
			w[j,j] = (dkn0 + dkn1) / 3.
		delta = delta.tocsr()
		w = w.tocsr()

		dk = (delta.T @ w) @ delta
		diff[imp0[k]:imp0[k+1], imp0[k]:imp0[k+1]] = dk / lambdas[k]

	H = diff[imp0[0]:,imp0[0]:]

	return H


def data_correlation(shotdat, icorrE, T0, sigma_t, sigma_m):
	"""
	Calculate the covariance matrix for data.

	Parameters
	----------
	shotdat : DataFrame
		GNSS-A shot dataset.
	icorrE : bool
		if the matrix has finite covariance terms or not.
	T0 : float
		Typical travel time (in sec.).
	sigma_t : float
		Correlation length (in sec.).
	sigma_m : float
		Ratio of correlation between the different transponders.

	Returns
	-------
	Ei : ndarray
		Inverse covariance matrix for data.
	logdetEi : float
		The value of log(|Ei|).
		|Ei| is the determinant of Ei.
	"""

	TT0 = shotdat.TT.values / T0

	if icorrE:
		ndata = shotdat.index.size
		sts = shotdat.ST.values
		mtids = shotdat.mtid.values

		E = lil_matrix( (ndata, ndata) )
		for i, (iMT, iST) in enumerate(zip( mtids, sts )):
			idx = shotdat[ ( abs(sts - iST) < sigma_t * 4.)].index
			dshot = np.abs(iST - sts[idx])/sigma_t
			dcorr = np.exp(-dshot) * (sigma_m + (1.-sigma_m)*(iMT==mtids[idx]))
			E[i,idx] = dcorr / TT0[i] / TT0[idx]
		E = E.tocsc()

		lu = linalg.splu(E)
		Ei = csc_matrix(lu.solve(np.eye(ndata)))

		diagL = lu.L.diagonal()
		diagU = lu.U.diagonal()
		diagL = diagL.astype(np.complex128)
		diagU = diagU.astype(np.complex128)
		logdetE  = np.log(diagL).sum() + np.log(diagU).sum()
		logdetEi = -logdetE.real

	else:
		Ei = csc_matrix( np.diag(TT0**2.) )
		logdetEi = (np.log(TT0**2.)).sum()

	return Ei, logdetEi
