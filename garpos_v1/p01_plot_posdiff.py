#!/usr/bin/python3
# -*- coding: utf-8 -*-
############################################
import numpy as np
import pandas as pd
import datetime, os
import matplotlib.pyplot as plt
from module.itrf_trans_posdiff import itrf_trans

sitelist  = ["KAMN", "KAMS", "MYGI", "MYGW", "FUKU", "CHOS", "BOSN", "SAGA" ]
sitelist += ["TOK1", "TOK2", "TOK3", "ZENW", "KUM1", "KUM2", "KUM3", "KUM4" ]
sitelist += ["SIOW", "SIO2", "MRT1", "MRT2", "MRT3", "TOS1", "TOS2", "ASZ1", "ASZ2", "HYG1", "HYG2" ]

#sitelist = ["ASZ2" ]
tag  = [ "%02d" % (i+1) for i in range(len(sitelist))]

#sitelist += ["TU08", "TU10", "TU12", "TU14", "TU17" ]
#tag += ["A","B","C","D","E"]

de = np.array([])
dn = np.array([])
du = np.array([])

#o = open("offset.dat","w")

suf = ""
#suf = "-nonKF"
#kf = "kf" # Kalman filter
#kf = "nonKF" # Kalman filter
kf = "sm" # Kalman smoothing

for k,site in enumerate(sitelist):
	
	rng = 0.4
	rng = 0.3
	
	dr = "result-bsp333"
	
	figname = dr + "/posdiff"+suf+"."+kf+"."+tag[k]+"."+site+".png"
	
	odt = "diff.TS."+site+".RTKLIBvsIT.dat"
	sgobs4 = "../posdiff-sgobs4/pos.diff.table." + site + ".multi_f"
	garpos = dr + "/" + site + "/posdiff-"+kf+".dat"
	#non_kf = "./result-nonKF/" + site + "/posdiff-nonKF.dat"
	
	if not os.path.exists(sgobs4):
		print("%s is not caclculated... " % sgobs4)
		continue
	if not os.path.exists(garpos):
		print("%s is not caclculated... " % garpos)
		continue
	
	print(garpos, figname)
	
	fm = '%m/%d/%Y'
	parser = lambda d: pd.datetime.strptime(d,fm)
	cnames = ['date','e','n','u','de','dn','du']
	ucols  = [0,1,2,3,4,5,6]
	
	pdsgobs4 = pd.read_csv(sgobs4, comment='#', usecols=ucols, names=cnames, index_col=0, parse_dates={'t':['date']}, date_parser=parser)
	pdsgobs4["year"] = [ x.year+((x-datetime.datetime(x.year,1,1,0,0,0)).days+1)/365. for x in pdsgobs4.index ]
	
	pdgarpos = pd.read_csv(garpos, comment='#', usecols=ucols, names=cnames, index_col=0, delim_whitespace=True)
	pdgarpos["year"] = pdgarpos.index
	
	trans = np.array([ itrf_trans(time, site)[1] for time in pdgarpos.index ])
	pdgarpos.e += trans[:,0]
	pdgarpos.n += trans[:,1]
	pdgarpos.u += trans[:,2]
	
	#pdnon_kf = pd.read_csv(non_kf, comment='#', usecols=ucols, names=cnames, index_col=0, delim_whitespace=True)
	#pdnon_kf["year"] = pdnon_kf.index
	#
	#trans = np.array([ itrf_trans(time, site)[1] for time in pdnon_kf.index ])
	#pdnon_kf.e += trans[:,0]
	#pdnon_kf.n += trans[:,1]
	#pdnon_kf.u += trans[:,2]
	
	dorigin = []
	diffmtx = np.array([])
	for n,i in enumerate(["e","n","u"]):
		dorigin.append(0.)
		pdsgobs4[i] -= pdsgobs4[i].mean()
		pdgarpos[i] -= pdgarpos[i].mean()
		#pdnon_kf[i] -= pdnon_kf[i].mean()
		diffmtx = np.append(diffmtx, abs(pdsgobs4[i].values))
		diffmtx = np.append(diffmtx, abs(pdgarpos[i].values))
		#diffmtx = np.append(diffmtx, abs(pdnon_kf[i].values))
	
	dmax = (int(np.max(diffmtx)*10.)+1)/10.
	rng = max(rng, dmax)
	#o.write(site + " " + " ".join(["%8.5f" % x for x in dorigin]) + "\n")
	
	# plot
	#syr = datetime.datetime(pdsgobs4.index[0].year   , 1, 1)
	#fyr = datetime.datetime(pdsgobs4.index[-1].year+1, 1, 1)
	syr = int(pdgarpos["year"].min())
	fyr = int(pdgarpos["year"].max())+1
	
	#title = tag[k] + " Seafloor displacement at " + site + "\n"
	title = " Seafloor displacement at " + site + "\n"
	fig = plt.figure(figsize = (8, 12))
	
	ax1 = fig.add_subplot(3, 1, 1)
	plt.tick_params(labelsize=14)
	plt.title(title, fontsize=20)
	ax2 = fig.add_subplot(3, 1, 2)
	plt.tick_params(labelsize=14)
	ax3 = fig.add_subplot(3, 1, 3)
	plt.tick_params(labelsize=14)
	ax1.set_ylim(-rng, rng)
	ax2.set_ylim(-rng, rng)
	ax3.set_ylim(-rng, rng)
	ax1.set_xlim(syr, fyr)
	ax2.set_xlim(syr, fyr)
	ax3.set_xlim(syr, fyr)
	ax1.set_ylabel("Eastward [m]", fontsize=16)
	ax2.set_ylabel("Northward [m]", fontsize=16)
	ax3.set_ylabel("Upward [m]", fontsize=16)
	ax3.set_xlabel("Year", fontsize=16)
	ax1.set_xticks( np.arange(syr, fyr, 1) )
	ax2.set_xticks( np.arange(syr, fyr, 1) )
	ax3.set_xticks( np.arange(syr, fyr, 1) )
	ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "'{:02d}".format(int(x)-2000)))
	ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "'{:02d}".format(int(x)-2000)))
	ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "'{:02d}".format(int(x)-2000)))
	ax1.grid()
	ax2.grid()
	ax3.grid()
	
	# sgobs
	ax1.scatter(pdsgobs4.year, pdsgobs4.e, marker="D", zorder=1, color = "navy", label="SGOBS v4.1.1 w/ IT")
	ax2.scatter(pdsgobs4.year, pdsgobs4.n, marker="D", zorder=1, color = "navy")
	ax3.scatter(pdsgobs4.year, pdsgobs4.u, marker="D", zorder=1, color = "navy")
	# garpos
	ax1.errorbar(pdgarpos.year, pdgarpos.e, yerr = pdgarpos.de, marker="o", zorder=2, linestyle='None', capsize=3, color = "red", label="GARPOS w/ PPP")
	ax2.errorbar(pdgarpos.year, pdgarpos.n, yerr = pdgarpos.dn, marker="o", zorder=2, linestyle='None', capsize=3, color = "red")
	ax3.errorbar(pdgarpos.year, pdgarpos.u, yerr = pdgarpos.du, marker="o", zorder=2, linestyle='None', capsize=3, color = "red")
	# nonKF
	#ax1.scatter(pdnon_kf.year, pdnon_kf.e, marker="x", zorder=3, color = "black", label="GARPOS non-KF")
	#ax2.scatter(pdnon_kf.year, pdnon_kf.n, marker="x", zorder=3, color = "black")
	#ax3.scatter(pdnon_kf.year, pdnon_kf.u, marker="x", zorder=3, color = "black")
	
	ax1.legend(fontsize=12)
	
	plt.savefig(figname)
	plt.close()
	

exit()

