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
from .map_estimation import MAPestimate


def parallelrun(inplist, maxcore):
	"""
	Run the MAP estimation in parallel.

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
	h2 = inplist.sig_t.values
	h3 = inplist.sig_m.values

	# Inversion design parameters
	p0 = inplist.loc[:, ["de","dn","du"]].values

	inp = list(zip(i0,i1,o1,o2,h0,h1,h2,h3,p0))

#	with Pool(processes=mc) as p:
#		reslist = p.starmap(MAPestimate, inp)
#		p.close()
	reslist = MAPestimate(inp[0][0],inp[0][1],inp[0][2],inp[0][3],inp[0][4],inp[0][5],inp[0][6],inp[0][7],inp[0][8])

	inplist["resfile"] = [ reslist[0] ]
	inplist["RMS_dTT"] = [ reslist[1]/1000. ]
	inplist["ABIC"] = [ reslist[2] ]
	inplist["dE"] = [ reslist[3][0] ]
	inplist["dN"] = [ reslist[3][1] ]
	inplist["dU"] = [ reslist[3][2] ]
#	inplist["resfile"] = [ r[0] for r in reslist ]
#	inplist["RMS_dTT"] = [ r[1]/1000. for r in reslist ]
#	inplist["ABIC"] = [ r[2] for r in reslist ]
#	inplist["dE"] = [ r[3][0] for r in reslist ]
#	inplist["dN"] = [ r[3][1] for r in reslist ]
#	inplist["dU"] = [ r[3][2] for r in reslist ]

	return inplist


def plot_hpres(df, resdir, filebase, score):
	"""
	Plot the results of hyperparameter search.

	Parameters
	----------
	df : DataFrame
		Result list for all models.
	resdir : string
		Directory to store the results.
	filebase : string
		File basename.
	score : string
		score for "goodness" of models ("ABIC" is used).

	Returns
	-------
	None (PNG file will be created.)
	"""


	minsore = min(df.loc[:,score])
	dfbest = df.loc[(df.loc[:,score] == minsore), :]

	df.loc[:,'dE'] -= dfbest['dE'].values[0]
	df.loc[:,'dN'] -= dfbest['dN'].values[0]
	df.loc[:,'dU'] -= dfbest['dU'].values[0]
	df.loc[:,'dE'] = df.loc[:,'dE'] * 100.
	df.loc[:,'dN'] = df.loc[:,'dN'] * 100.
	df.loc[:,'dU'] = df.loc[:,'dU'] * 100.
	df.iloc[::-1]

	fs = 10
	axs = scatter_matrix(df.iloc[:,0:], figsize=(fs,fs), alpha=0.4,
						 c=list(-df.loc[:, score]), diagonal='kde')

	title = "%s" % filebase
	plt.suptitle(title, y=0.95)

	png = resdir + "search-%s.png" % filebase
	print(png)
	plt.savefig(png)
	plt.close()

	return


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
	sig_ts = icfg.get("HyperParameters", "sigma_t").split()
	sig_ms = icfg.get("HyperParameters", "sigma_mt").split()

	lamb0s = np.array(list(map(float, lamb0s)))
	glambs = np.array(list(map(float, glambs)))
	sig_ts = np.array(list(map(float, sig_ts)))
	sig_ms = np.array(list(map(float, sig_ms)))

	nl = len(lamb0s)
	ng = len(glambs)
	nt = len(sig_ts)
	nm = len(sig_ms)
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
	hps = np.array(list(itertools.product(lamb0s, glambs, sig_ts, sig_ms)))

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
	inputs['sig_t'] = hps[:,2]*60.
	inputs['sig_m'] = hps[:,3]

	print(inputs)

	inputs["de"] = 0.0
	inputs["dn"] = 0.0
	inputs["du"] = 0.0
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
		df["sigma(t)"] = df.sig_t/60.

		df = df.loc[:,[score,"log(Lambda)","log(L_grad)","sigma(t)","dE","dN","dU"]]

		if nl  <= 1:
			df=df.drop("log(Lambda)", axis=1)
		if ng <= 1:
			df=df.drop("log(L_grad)", axis=1)
		if nt  <= 1:
			df=df.drop("sigma(t)", axis=1)

		print(df)
		of = wkdir + "searchres-%s.dat" % filebase
		df.to_csv(of)

		plot_hpres(df, outdir, filebase, score)

	return resf
