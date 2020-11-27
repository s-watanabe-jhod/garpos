#!/usr/bin/python_ana3
# -*- coding: utf-8 -*-
############################################
import numpy as np
import pandas as pd
import shutil
import datetime
import os
import matplotlib.pyplot as plt
#from garpos.itrf_trans_posdiff import itrf_trans
import coord_trans

sitelist  = ["KAMN", "KAMS", "MYGI", "MYGW", "FUKU", "CHOS", "BOSN", "SAGA" ]
sitelist += ["TOK1", "TOK2", "TOK3", "ZENW", "KUM1", "KUM2", "KUM3", "KUM4" ]
sitelist += ["SIOW", "SIO2", "MRT1", "MRT2", "MRT3", "TOS1", "TOS2", "ASZ1", "ASZ2", "HYG1", "HYG2" ]

#sitelist = ["SIOW" ]
tag  = [ "%02d" % (i+1) for i in range(len(sitelist))]

sitelist += ["TU08", "TU10", "TU12", "TU14", "TU17" ]
tag += ["A","B","C","D","E"]

de = np.array([])
dn = np.array([])
du = np.array([])

figdir = "timeseries-mgr/"
figext = ".png"
suf = ""
kf = "fix" # Kalman smoothing

if kf == "kf":
	plotlist0 = pd.DataFrame( ["Garpos-KF", "Garpos-FIX", "SGOBS-v4.1.1(IT)"], columns = ['label'] )
elif kf == "sm":
	plotlist0 = pd.DataFrame( ["Garpos-KS", "Garpos-FIX", "SGOBS-v4.1.1(IT)"], columns = ['label'] )
elif kf == "fix":
	plotlist0 = pd.DataFrame( ["GARPOS", "Garpos-FIX", "SGOBS-v4.0.2"], columns = ['label'] )
	#plotlist0 = pd.DataFrame( ["Garpos-FIX", "Garpos-FIX-2gen", "SGOBS-v4.0.2(PPP)"], columns = ['label'] )
else:
	print("error")
	exit()
#plotlist0["pdfile"] = ["./fix-abic/res.SITE.dat", "none", "../posdiff-sgobs4/pos.diff.table.SITE.multi_f"]
#plotlist0["pdfile"] = ["./fix-abic/res.SITE.dat", "none", "../posdiff-sgobs-sciadv/pos.diff.table.SITE.multi_f"]
plotlist0["pdfile"] = ["./fix-abic/res.SITE.dat", "none", "../posdiff-sgobs-mgr/pos.diff.table.SITE.multi_rtk"]

plotlist0["iplot"]  = [ True, True, True ]
plotlist0["color"]  = [ "orange", "red", "navy" ]
plotlist0["marker"] = [ "o", "x", "D" ]
plotlist0["zorder"] = [ 2, 3, 1 ]
plotlist0["yerr"]   = [ False, False, False ]
ylabels = [ "Eastward [m]", "Northward [m]", "Upward [m]"]

if not os.path.exists(figdir+"/"):
	os.makedirs(figdir)

for k,site in enumerate(sitelist):
	
	rng = 0.3
	
	plist = plotlist0.copy()
	figname = figdir + "posdiff"+suf+"."+kf+"."+tag[k]+"."+site+figext
	
	data = [""] * len(plist)
	dorigin = [0.] * len(plist)
	diffmtx = np.array(dorigin)
	sflag = False
	
	for i in range(len(plist)):
		
		if not plist.loc[i,"iplot"]:
			continue
		
		#plist.loc[i,"pdfile"] = plist.loc[i,"pdfile"].replace("SITE", site)
		fl = plist.loc[i,"pdfile"].replace("SITE", site)
		
		if (not os.path.exists(fl)) or os.path.getsize(fl) < 10:
			print("%s is not caclculated... " % fl)
			plist.loc[i,"iplot"] = False
			if i == 0:
				sflag = True
			continue
		
		if not "sgobs" in fl:
			shutil.copy(fl, figdir)
		
		ucols  = [0,1,2,3,4,5,6]
		cnames = ['date','e','n','u','de','dn','du']
		
		if "sgobs" in fl:
			fm = '%m/%d/%Y'
			parser = lambda d: pd.datetime.strptime(d,fm)
			df = pd.read_csv(fl, comment='#', usecols=ucols, names=cnames, index_col=0, parse_dates={'t':['date']}, date_parser=parser)
			df["year"] = [ x.year+((x-datetime.datetime(x.year,1,1,0,0,0)).days+1)/365. for x in df.index ]
			df.set_index("year", drop=False,inplace=True)
		else:
			df = pd.read_csv(fl, comment='#', usecols=ucols, names=cnames, index_col=0, delim_whitespace=True)
			df["year"] = df.index
		
		if True:
			trans = np.array([ coord_trans.itrf_trans_posdiff(t, site)[1] for t in df.index ])
			df.e += trans[:,0]
			df.n += trans[:,1]
			df.u += trans[:,2]
		"""
		if i == 0:
			print(df.loc[ (df.year < 2014.9) ,["year"]])
			df.loc[(df.year < 2014.9),["e"]] += 0.08
			df.loc[(df.year < 2014.9),["n"]] += 0.00
		"""
		
		for n, j in enumerate(["e","n","u"]):
			df[j] -= df[j].mean()
			diffmtx[i] = max(diffmtx[i], np.max(abs(df[j].values)))
		
		data[i] = df
	
	if sflag:
		continue
	
	print(figname)
	
	# plot range
	dmax = (int((np.max(diffmtx)+0.03)*10.) + 1)/10.
	rng = max(rng, dmax)
	syr = int(data[0]["year"].min())
	fyr = int(data[0]["year"].max())+1
	
	# set plot space
	title = "(" + tag[k] + ") Seafloor displacement at " + site + "\n"
	#title = " Seafloor displacement at " + site + "\n"
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
	ax1.set_ylabel(ylabels[0], fontsize=16)
	ax2.set_ylabel(ylabels[1], fontsize=16)
	ax3.set_ylabel(ylabels[2], fontsize=16)
	ax3.set_xlabel("Year", fontsize=16)
	ax1.set_xticks( np.arange(syr, fyr, 1) )
	ax2.set_xticks( np.arange(syr, fyr, 1) )
	ax3.set_xticks( np.arange(syr, fyr, 1) )
	ax1.axes.xaxis.set_ticklabels([])
	ax2.axes.xaxis.set_ticklabels([])
	ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "'{:02d}".format(int(x)-2000)))
	ax1.grid()
	ax2.grid()
	ax3.grid()
	
	for i, df in enumerate(data):
		
		if not plist.loc[i,"iplot"]:
			continue
		
		lb = plist.loc[i,"label"]
		mk = plist.loc[i,"marker"]
		cl = plist.loc[i,"color"]
		zo = plist.loc[i,"zorder"]
		
		if plist.loc[i,"yerr"]:
			ax1.errorbar(df.year, df.e, yerr = df.de*3., marker=mk, zorder=zo, linestyle='None', capsize=3, color=cl, label=lb)
			ax2.errorbar(df.year, df.n, yerr = df.dn*3., marker=mk, zorder=zo, linestyle='None', capsize=3, color=cl)
			ax3.errorbar(df.year, df.u, yerr = df.du*3., marker=mk, zorder=zo, linestyle='None', capsize=3, color=cl)
			
		else:
			ax1.scatter(df.year, df.e, marker=mk, zorder=zo, color=cl, label=lb)
			ax2.scatter(df.year, df.n, marker=mk, zorder=zo, color=cl)
			ax3.scatter(df.year, df.u, marker=mk, zorder=zo, color=cl)
		
	
	ax1.legend(fontsize=12)
	plt.savefig(figname)
	plt.close()
	

exit()

