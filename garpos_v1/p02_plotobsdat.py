#!/usr/bin/python3
# -*- coding: utf-8 -*-
############################################
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import lombscargle as LS
import pandas as pd
import math, datetime, os, glob, configparser
from module.utilities import *
from pandas import plotting 

sitelist = ["KUM3"]
#sitelist  = ["KAMN", "KAMS", "MYGI", "MYGW", "FUKU", "CHOS", "BOSN", "SAGA" ]
#sitelist += ["TOK1", "TOK2", "TOK3", "ZENW", "KUM1", "KUM2", "KUM3", "KUM4" ]
#sitelist += ["SIOW", "SIO2", "MRT1", "MRT2", "MRT3", "TOS1", "TOS2", "ASZ1", "ASZ2", "HYG1", "HYG2" ]
#sitelist += ["TOK1", "TOK2", "TOK3", "KUM1", "KUM2", "KUM3" ]
#sitelist += ["SIOW", "MRT1", "MRT2", "TOS1", "TOS2", "ASZ1", "ASZ2", "HYG1", "HYG2" ]

outdir    = "pngtmp/"

for site in sitelist:
	#targetdir = "./result-kf/????/"
	targetdir = "./result-bsp333/%s/" % site
	#targetdir = "./result-bic/ASZ2/"
	#targetdir = "./tidefree/result-deg04g/"

	mkdirectory(outdir)

	obsdata = sorted(glob.glob(targetdir+"*-obs.csv"))
	resdata = sorted(glob.glob(targetdir+"*-res.dat"))

	df = pd.DataFrame(np.zeros(len(obsdata)), columns = ['ep'])
	i = 0

	for fl, rf in zip(obsdata,resdata):
		epname = os.path.basename(fl).split("-")[0]
		outf = outdir+epname+".png"
		#print(fl, outf)
		from pandas import plotting 

		shotdat = pd.read_csv(fl, comment='#', index_col=0)
		shotdat = shotdat[~shotdat.flag]
		
		mall = shotdat.ResiTT.mean()*1000.
		m15 = shotdat[ (shotdat.TakeOff < 15.) ].ResiTT.mean()*1000.
		m10 = shotdat[ (shotdat.TakeOff < 10.) ].ResiTT.mean()*1000.
		m45 = shotdat[ (shotdat.TakeOff > 45.) ].ResiTT.mean()*1000.
		m50 = shotdat[ (shotdat.TakeOff > 50.) ].ResiTT.mean()*1000.
		
		n15 = len(shotdat[ (shotdat.TakeOff < 15.) ].index)
		n10 = len(shotdat[ (shotdat.TakeOff < 10.) ].index)
		n45 = len(shotdat[ (shotdat.TakeOff > 45.) ].index)
		n50 = len(shotdat[ (shotdat.TakeOff > 50.) ].index)
		
		df.loc[i, "ep"] = i
		df.loc[i, "allTA"] = mall
		df.loc[i, "deg10"] = m10 - mall
		df.loc[i, "deg15"] = m15 - mall
		df.loc[i, "deg45"] = m45 - mall
		df.loc[i, "deg50"] = m50 - mall
		df.loc[i, "num10"] = n10
		df.loc[i, "num15"] = n15
		df.loc[i, "num45"] = n45
		df.loc[i, "num50"] = n50
		print(fl)
		print(os.path.basename(fl))
		df.loc[i, "file"] = os.path.basename(fl)
		
		cfg = configparser.ConfigParser()
		cfg.read(rf, 'UTF-8')
		dcnt = cfg.get("Model-parameter", "dCentPos").split()
		de = float(dcnt[0])
		dn = float(dcnt[1])
		du = float(dcnt[2])
		se = float(dcnt[3])
		sn = float(dcnt[4])
		su = float(dcnt[5])
		
		df.loc[i, "de"] = de
		df.loc[i, "dn"] = dn
		df.loc[i, "du"] = du
		
		
		#print(" %12.7f %12.7f %12.7f %12.7f %4d %4d %4d %s" % (mall, m10, m15, m45, n10, n15, n45, os.path.basename(fl)) )
		i += 1
		
		#plt.title(epname)
		#plt.scatter(shotdat.TakeOff, shotdat.ResiTT*1000.)
		#plt.grid()
		##plt.savefig(outf)
		#plt.show()
		#plt.close()
		#
		#exit()
	
	#print(df)
	
	# not show RMS
	df = df.drop("ep", axis=1)
	df = df.drop("file", axis=1)
	df = df.drop("num10", axis=1)
	df = df.drop("num15", axis=1)
	df = df.drop("num45", axis=1)
	df = df.drop("num50", axis=1)
	df = df.drop("deg15", axis=1)
	df = df.drop("deg45", axis=1)
	df = df.drop("allTA", axis=1)
	
	df.loc[:,'de'] = df.loc[:,'de'] * 100.
	df.loc[:,'dn'] = df.loc[:,'dn'] * 100.
	df.loc[:,'du'] = df.loc[:,'du'] * 100.
	
	df = df[~ df.isnull().any(axis=1)]
	#print(df)
	
	fs = 10
	axs = plotting.scatter_matrix(df.iloc[:, 0:], figsize=(fs, fs), c=list(-df.index), diagonal='kde', alpha=0.4)
	
	title = "%s" % site
	plt.suptitle(title, y=0.95)
	
	"""
	r = 3.5
	nnn = 1
	n0m = int(df.imp0.max()) + 1
	n1m = int(df.imp1.max()) + 1
	ncolumn = len(df.columns)
	for i in range(ncolumn):
		for j in range(ncolumn):
			ax = axs[i,j]
			ax.yaxis.set_label_coords(-0.5, 0.5)
			ax.xaxis.set_label_coords(0.5, -0.5)
			
			# set xrange
			rjmax = max(df.iloc[:,j])
			rjmin = min(df.iloc[:,j])
			if r > max(abs(rjmax),abs(rjmin)):
				if j > nnn:
					ax.set_xlim([-r,r])
			if j == nnn:
				ax.set_xlim([0,n0m])
			elif j == nnn+1:
				ax.set_xlim([-1,n1m])
			
			if i == j:
				continue
			
			# set yrange
			rimax = max(df.iloc[:,i])
			rimin = min(df.iloc[:,i])
			if r > max(abs(rimax),abs(rimin)):
				if i > nnn:
					ax.set_ylim([-r,r])
			if i == nnn:
				ax.set_ylim([0,n0m])
			elif i == nnn+1:
				ax.set_ylim([-1,n1m])

	bicpng = resdir + "bic-%s.png" % sitecamp
	print(bicpng)
	"""
	png = outdir + "/TOangle-%s.png" % site
	plt.savefig(png)
	#plt.show()
	plt.close()
	
	ave10 = df.deg10.mean()
	ave50 = df.deg50.mean()
	
	cor = df.corr()
	
	print(" %10.5f %10.5f %s" % (ave10, ave50, site))
	print(cor)







exit()

plt.scatter(df.ep, df.deg10)
plt.scatter(df.ep, df.deg15)
#plt.scatter(df.ep, df.deg45)
plt.scatter(df.ep, df.deg50)
plt.legend()
plt.show()
plt.close()

"""
plt.scatter(df.deg10, df.deg15)
plt.legend()
plt.show()
"""
plt.scatter(df.ep, df.deg15-df.deg45)
plt.scatter(df.ep, df.du)
plt.legend()
plt.show()
