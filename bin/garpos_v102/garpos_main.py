"""
Created:
	07/01/2020 by S. Watanabe
Contains:
	parallelrun
	plot_hpres
	drive_garpos
"""
import os
import glob
import math
import shutil
import configparser
from multiprocessing import Pool
import itertools
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# garpos module
from .mp_estimation import MPestimate


def parallelrun(inplist, maxcore):
	"""
	Run the model parameter estimation in parallel.

	Parameters
	----------
	inplist : DataFrame
		List of arguments for the function.
	maxcore : int
		maximum number of parallelization.

	Returns
	-------
	inplist : DataFrame
		List of arguments for the function in which brief results are added.
	"""

	npara = len(inplist.index)
	mc = min(maxcore, npara)

	# Input files
	i0 = inplist.cfgfile
	i1 = inplist.invcfg

	# Output parameters
	o1 = inplist.outdir
	o2 = inplist.suffix

	# Hyperparameters
	h0 = inplist.lamb0.values
	h1 = inplist.lgrad.values
	h2 = inplist.mu_t.values
	h3 = inplist.mu_m.values

	inp = list(zip(i0,i1,o1,o2,h0,h1,h2,h3))

	with Pool(processes=mc) as p:
		reslist = p.starmap(MPestimate, inp)
		p.close()

	inplist["resfile"] = [ r[0] for r in reslist ]
	inplist["RMS_dTT"] = [ r[1]/1000. for r in reslist ]
	inplist["ABIC"] = [ r[2] for r in reslist ]
	inplist["dE"] = [ r[3][0] for r in reslist ]
	inplist["dN"] = [ r[3][1] for r in reslist ]
	inplist["dU"] = [ r[3][2] for r in reslist ]

	return inplist


def drive_garpos(cfgf, icfgf, outdir, suf, maxcore):
	"""
	Main driver to run GARPOS.

	Parameters
	----------
	cfgf : string
		Path to the site-parameter file.
	icfgf : stirng
		Path to the analysis-setting file.
	outdir : string
		Directory to store the results.
	suf : string
		Suffix to be added for result files.
	maxcore : int
		maximum number of parallelization.

	Returns
	-------
	resf : string
		Result site-paramter file name (min-ABIC model).
	"""

	# Set Hyperparamters for search
	icfg = configparser.ConfigParser()
	icfg.read(icfgf, 'UTF-8')

	lamb0s = icfg.get("HyperParameters", "Log_Lambda0").split()
	glambs = icfg.get("HyperParameters", "Log_gradLambda").split()
	mu_ts = icfg.get("HyperParameters", "mu_t").split()
	mu_ms = icfg.get("HyperParameters", "mu_mt").split()

	lamb0s = np.array(list(map(float, lamb0s)))
	glambs = np.array(list(map(float, glambs)))
	mu_ts = np.array(list(map(float, mu_ts)))
	mu_ms = np.array(list(map(float, mu_ms)))

	nl = len(lamb0s)
	ng = len(glambs)
	nt = len(mu_ts)
	nm = len(mu_ms)
	nmodels = nl * ng * nt * nm

	if nmodels == 1:
		wkdir = outdir
	elif nmodels > 1:
		wkdir  = outdir+ "/lambda/"
	else:
		print("error in hyper paramter setting")
		sys.exit(1)

	if not os.path.exists(wkdir+"/"):
		os.makedirs(wkdir)

	# Set File Name
	cfg = configparser.ConfigParser()
	cfg.read(cfgf, 'UTF-8')
	site = cfg.get("Obs-parameter", "Site_name")
	camp = cfg.get("Obs-parameter", "Campaign")
	filebase = site + "." + camp + suf

	# Set Input parameter list for ParallelRun
	hps = np.array(list(itertools.product(lamb0s, glambs, mu_ts, mu_ms)))

	sufs = [ suf ] * nmodels
	for i, hp in enumerate(hps):
		if nl > 1:
			sufs[i] += "_L%+05.1f" % hp[0]
		if ng > 1:
			sufs[i] += "_g%+05.1f" % hp[1]
		if nt > 1:
			sufs[i] += "_T%03.1f" % hp[2]
		if nm > 1:
			sufs[i] += "_mt%03.1f" % hp[3]

	inputs = pd.DataFrame(sufs, columns = ['suffix'])

	inputs['lamb0'] = 10.**hps[:,0]
	inputs['lgrad'] = 10.**hps[:,1]
	inputs['mu_t'] = hps[:,2]*60.
	inputs['mu_m'] = hps[:,3]

	print(inputs)

	inputs["cfgfile"] = cfgf
	inputs["invcfg"] = icfgf
	inputs["outdir"] = wkdir

	outputs = parallelrun(inputs, maxcore)

	resf = outputs.resfile[0]
	score='ABIC'

	df = outputs.sort_values(score, ascending=True).reset_index(drop=True)
	resf = df.resfile[0]

	if nmodels > 1:
		print(resf)
		bestfile = os.path.basename(resf)
		dfl = os.path.abspath(resf)
		dfl = os.path.dirname(dfl)
		fls = sorted(glob.glob(dfl+"/"+bestfile.replace("-res.dat","-*")))
		for ff in fls:
			shutil.copy(ff, outdir+"/"+os.path.basename(ff))

		# to summarize the results
		df["log(Lambda)"] = [math.log10(l) for l in df.lamb0]
		df["log(L_grad)"] = [math.log10(l) for l in df.lgrad]
		df["mu(t)"] = df.mu_t/60.

		df = df.loc[:,[score,"log(Lambda)","log(L_grad)","mu(t)","dE","dN","dU"]]

		if nl  <= 1:
			df=df.drop("log(Lambda)", axis=1)
		if ng <= 1:
			df=df.drop("log(L_grad)", axis=1)
		if nt  <= 1:
			df=df.drop("mu(t)", axis=1)

		print(df)
		of = wkdir + "searchres-%s.dat" % filebase
		df.to_csv(of)

	return resf
