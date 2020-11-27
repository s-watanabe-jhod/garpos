"""
Created:
	07/01/2020 by S. Watanabe
"""
import os
import sys
import math
import configparser
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, linalg
import pandas as pd

# garpos module
from .setup_model import init_position, make_knots, derivative2, data_correlation
from .forward import calc_forward, calc_gamma, jacobian_pos
from .output import outresults


def MPestimate(cfgf, icfgf, odir, suf, lamb0, lgrad, mu_t, mu_m, denu):
	"""
	Run the model parameter estimation. (under given hyperparameter)
	
	Parameters
	----------
	cfgf : string
		Path to the site-parameter file.
	icfgf : stirng
		Path to the analysis-setting file.
	odir : string
		Directory to store the results.
	suf : string
		Suffix to be added for result files.
	lamb0 : float
		Hyperparameter (Lambda_0)^2.
		Controls the smoothness for gamma.
	lgrad : float
		Hyperparameter (Lambda_g/Lambda_0)^2.
		Controls the smoothness for gradient-components of gamma.
	mu_t : float
		Hyperparameter (mu_t).
		Controls the correlation length for acoustic data.
	mu_m : float
		Hyperparameter (mu_MT).
		Controls the inter-transponder data correlation.
	denu : ndarray (len=3)
		Positional offset (only applicable in case invtyp = 1).
	
	Returns
	-------
	resf : string
		Result site-parameter file name (min-ABIC model).
	datarms : float
		RMS for "real" travel time (NOTE: not in log).
	abic : float
		ABIC value.
	dcpos : ndarray
		Positional difference of array center and its variances.
	"""
	
	spdeg = 3
	np.set_printoptions(threshold=np.inf)
	
	if lamb0 <= 0.:
		print("Lambda must be > 0")
		sys.exit(1)
	
	################################
	### Set Inversion Parameters ###
	################################
	icfg = configparser.ConfigParser()
	icfg.read(icfgf, 'UTF-8')
	invtyp = int(icfg.get("Inv-parameter","inversiontype"))
	nmp0 = int(icfg.get("Inv-parameter","nmp0"))
	nmp1 = int(icfg.get("Inv-parameter","nmp1"))
	nmp2 = int(icfg.get("Inv-parameter","nmp2"))
	rsig = float(icfg.get("Inv-parameter","RejectCriteria"))
	scale = float(icfg.get("Inv-parameter","traveltimescale"))
	maxloop = int(icfg.get("Inv-parameter","maxloop"))
	ConvCriteria = float(icfg.get("Inv-parameter","ConvCriteria"))
	
	if invtyp == 0:
		nmp0 = 0
		nmp1 = 0
		nmp2 = 0
	if nmp0+nmp1+nmp2 == 0:
		invtyp = 0
	
	#############################
	### Set Config Parameters ###
	#############################
	cfg = configparser.ConfigParser()
	cfg.read(cfgf, 'UTF-8')
	
	### Read obs file ###
	obsfile  = cfg.get("Data-file", "datacsv")
	shots = pd.read_csv(obsfile, comment='#', index_col=0)
	
	# check NaN in shotdata
	shots = shots[~shots.isnull().any(axis=1)].reset_index(drop=True)
	
	### Sound speed profile ###
	svpf = cfg.get("Obs-parameter", "SoundSpeed")
	svp = pd.read_csv(svpf, comment='#')
	
	### IDs of existing transponder ###
	MTs  = cfg.get("Site-parameter", "Stations").split()
	MTs = [ str(mt) for mt in MTs ]
	nMT = len(MTs)
	shots = shots[ shots.MT.isin(MTs) ].reset_index(drop=True)
	
	############################
	### Set Model Parameters ###
	############################
	mode = "Inversion-type %1d" % invtyp
	mppos, Dipos, slvidx0, mtidx = init_position(cfg, denu, MTs)
	
	# in case where positions should not be solved
	if invtyp == 1:
		Dipos = lil_matrix( (0, 0) )
		slvidx0 = np.array([])
	
	nmppos = len(slvidx0)
	cnt = np.array([ mppos[imt*3:imt*3+3] for imt in range(nMT)])
	cnt = np.mean(cnt, axis=0)
	# MT index for model parameter
	shots['mtid'] = [ mtidx[mt] for mt in shots['MT'] ]
	
	### Set Model Parameters for gamma ###
	nmpsv = [nmp0, nmp1, nmp1, nmp2, nmp2]
	glambda = lamb0 * lgrad
	lambdas = [lamb0] +[lamb0 * lgrad]*4
	
	knots = make_knots(shots, spdeg, nmpsv)
	ncps = [ max([0, len(kn)-spdeg-1]) for kn in knots]
	
	# set pointers for model parameter vector
	imp0 = np.cumsum(np.array([len(mppos)] + ncps))
	
	# set full model parameter vector
	mp = np.zeros(imp0[-1])
	mp[:imp0[0]] = mppos
	
	slvidx = np.append(slvidx0, np.arange(imp0[0],imp0[-1],dtype=int))
	slvidx = slvidx.astype(int)
	H = derivative2(imp0, spdeg, knots, lambdas)
	
	### set model parameter to be estimated ###
	mp0 = mp[slvidx]
	mp1 = mp0.copy()
	nmp = len(mp0)
	
	### Set a priori covariance for model parameters ###
	Di = lil_matrix( (nmp, nmp) )
	Di[:nmppos,:nmppos] = Dipos
	Di[nmppos:,nmppos:] = H
	Di = Di.tocsc()
	
	rankDi = np.linalg.matrix_rank(Di.toarray())
	eigvDi = np.linalg.eigh(Di.toarray())[0]
	eigvDi = eigvDi[ np.where(np.abs(eigvDi.real) > 1.e-9/lamb0)].real
	if rankDi != len(eigvDi):
		print(eigvDi)
		print(np.linalg.matrix_rank(Di), len(eigvDi))
		print("Error in calculating eigen value of Di !!!")
		sys.exit(1)
	logdetDi = np.log(eigvDi).sum()
	
	# Initial parameters for gradient gamma
	shots['sta0_e'] = mp[shots['mtid']+0] + mp[len(MTs)*3+0]
	shots['sta0_n'] = mp[shots['mtid']+1] + mp[len(MTs)*3+1]
	shots['sta0_u'] = mp[shots['mtid']+2] + mp[len(MTs)*3+2]
	shots['mtde'] = (shots['sta0_e'].values - cnt[0])
	shots['mtdn'] = (shots['sta0_n'].values - cnt[1])
	shots['de0'] = shots['ant_e0'].values - shots['ant_e0'].values.mean()
	shots['dn0'] = shots['ant_n0'].values - shots['ant_n0'].values.mean()
	shots['de1'] = shots['ant_e1'].values - shots['ant_e1'].values.mean()
	shots['dn1'] = shots['ant_n1'].values - shots['ant_n1'].values.mean()
	
	#####################################
	# Set log(TT/T0) and initial values #
	#####################################
	# calc average depth*2 (characteristic length)
	L0 = np.array([(mp[i*3+2] + mp[nMT*3+2]) for i in range(nMT)]).mean()
	L0 = abs(L0 * 2.)
	
	# calc depth-averaged sound speed (characteristic length/time)
	vl = svp.speed.values
	dl = svp.depth.values
	avevlyr = [ (vl[i+1]+vl[i])*(dl[i+1]-dl[i])/2. for i in svp.index[:-1]]
	V0 = np.array(avevlyr).sum()/(dl[-1]-dl[0])
	
	# calc characteristic time
	T0 = L0 / V0
	shots["logTT"] = np.log(shots.TT.values/T0)
	
	######################
	## data correlation ##
	######################
	# Calc initial ResiTT
	if invtyp != 0:
		shots["gamma"] = 0.
	shots = calc_forward(shots, mp, nMT, icfg, svp, T0)
	
	# Set data covariance
	icorrE = rsig < 0.1 and mu_t > 1.e-3
	if not icorrE:
		mu_t = 0.
	tmp = shots[~shots['flag']].reset_index(drop=True).copy()
	scale = scale/T0
	Ei, logdetEi = data_correlation(tmp, icorrE, T0, mu_t, mu_m)
	Ei = Ei/scale**2.
	
	#############################
	### loop for Least Square ###
	#############################
	comment = ""
	iconv = 0
	
	for iloop in range(maxloop):
		
		# tmp contains unrejected data
		tmp = shots[~shots['flag']].reset_index(drop=True).copy()
		ndata = len(tmp.index)
		
		############################
		### Calc Jacobian matrix ###
		############################
		
		# Set array for Jacobian matrix
		if rsig > 0.1 or iloop == 0:
			jcb = lil_matrix( (nmp, ndata) )
		
		# Calc Jacobian for gamma
		if invtyp != 0 and (rsig > 0.1 or iloop == 0):
			mpj = np.zeros(imp0[5])
			imp = nmppos
			
			for impsv in range(imp0[0],imp0[-1]):
				mpj[impsv] = 1.
				gamma, a = calc_gamma(mpj, tmp, imp0, spdeg, knots)
				
				jcb[imp,:] = -gamma*scale
				imp += 1
				mpj[impsv] = 0.
		
		# Calc Jacobian for position
		if invtyp != 1:
			jcb0 = jacobian_pos(icfg, mp, slvidx0, tmp, mtidx, svp, T0)
			jcb[:nmppos, :] = jcb0[:nmppos, :]
		
		jcb = jcb.tocsc()
		
		############################
		### CALC model parameter ###
		############################
		AktEi = jcb @ Ei
		AktEiAk = AktEi @ jcb.T
		Cki = AktEiAk + Di
		rk  = AktEi @ tmp.ResiTT.values + Di @ (mp0-mp1)
		
		Ck = linalg.inv( Cki )
		alpha = 1.0 # fixed
		dmp  = alpha * ( Ck @ rk )
		dxmax = max(abs(dmp[:]))
		if invtyp == 1:
			dposmax = 0. # no loop needed in invtyp = 1
		else:
			dposmax = max(abs(dmp[:nmppos]))
			if dxmax > 10.:
				alpha = 10./dxmax
				dmp = alpha * dmp
				dxmax = max(abs(dmp[:]))
		
		mp1 += dmp # update mp1 (=x(k+1))
		for j in range(len(mp1)):
			mp[slvidx[j]] = mp1[j]
		
		####################
		### CALC Forward ###
		####################
		if invtyp != 0:
			gamma, a  = calc_gamma(mp, shots, imp0, spdeg, knots)
			shots["gamma"] = gamma * scale
			av = np.array(a) * scale * V0
		else:
			av = 0. # dummy
		shots['dV'] = shots.gamma * V0
		
		shots = calc_forward(shots, mp, nMT, icfg, svp, T0)
		tmp = shots[~shots['flag']].reset_index(drop=True).copy()
		ndata = len(tmp.index)
		if rsig > 0.1:
			Ei, logdetEi = data_correlation(tmp, icorrE, T0, mu_t, mu_m)
			Ei = Ei/scale**2.
		
		rttadp = tmp.ResiTT.values
		rttvec = csr_matrix( np.array([rttadp]) )
		misfit = ((rttvec @ Ei) @ rttvec.T)[0,0]
		
		# Calc Model-parameters' RMSs
		rms = lambda d: np.sqrt((d ** 2.).sum() / d.size)
		mprms   = rms( dmp )
		rkrms   = rms( rk  )
		datarms = rms(tmp.ResiTTreal.values)
		
		aved = np.array([(mp[i*3+2] + mp[nMT*3+2]) for i in range(nMT)]).mean()
		reject = shots[shots['flag']].index.size
		ratio  = 100. - float(reject)/float(len(shots.index))*100.
		
		##################
		### Check Conv ###
		##################
		loopres  = "%s Loop %2d-%2d, " % (mode, 1, iloop+1)
		loopres += "RMS(TT) = %10.6f ms, " % (datarms*1000.)
		loopres += "used_shot = %5.1f%%, reject = %4d, " % (ratio, reject)
		loopres += "Max(dX) = %10.4f, Hgt = %10.3f" % (dxmax, aved)
		print(loopres)
		comment += "#"+loopres+"\n"
		
		if dxmax < ConvCriteria/100. or dposmax < ConvCriteria/1000.:
			break
		elif dxmax < ConvCriteria:
			iconv += 1
			if iconv == 2:
				break
		else:
			iconv = 0
	
	#######################
	# calc ABIC and sigma #
	#######################
	dof = float(ndata + rankDi - nmp)
	S = misfit + ( (mp0-mp1) @ Di ) @ (mp0-mp1)
	
	eigvCki = np.linalg.eigh(Cki.toarray())[0].real
	logdetCki = np.log(eigvCki).sum()
	
	abic = dof * math.log(S) - logdetEi - logdetDi + logdetCki
	sigobs  = (S/dof)**0.5 * scale
	C = S/dof * Ck.toarray()
	rmsmisfit = (misfit/ndata) **0.5 * sigobs
	
	finalres  = " ABIC = %18.6f " % abic
	finalres += " misfit = % 6.3f " % (rmsmisfit*1000.)
	finalres += suf
	print(finalres)
	comment += "# " + finalres + "\n"
	
	comment += "# lambda_0^2 = %12.8f\n" % lamb0
	comment += "# lambda_g^2 = %12.8f\n" % (lamb0 * lgrad)
	comment += "# mu_t = %12.8f sec.\n" % mu_t
	comment += "# mu_MT = %5.4f\n" % mu_m
	
	#####################
	# Write Result data #
	#####################
	resf, dcpos = outresults(odir, suf, cfg, invtyp, imp0, slvidx0,
							 C, mp, shots, comment, MTs, mtidx, av)
	
	return [resf, datarms, abic, dcpos]
	
