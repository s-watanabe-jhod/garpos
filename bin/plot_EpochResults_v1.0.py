#!/usr/bin/env python

import os
import sys
import glob
import math
import datetime
from datetime import timedelta
import configparser
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def plot_residuals(resf, obsfile, mpfile, d0, MTs, V0, ext):
	
	figdir = os.path.dirname(resf)+"/fig"
	if not os.path.exists(figdir+"/"):
		os.makedirs(figdir)
	
	offsec = 60.*60.
	ngrad  = 6
	setcmap = plt.get_cmap("Set3")
	mtcmap  = plt.get_cmap("tab10")
	
	alldat = pd.read_csv(obsfile, comment='#', index_col=0)
	alldat['dt'] = [ d0 + timedelta(seconds=t) for t in alldat['ST'] ]
	
	setlist = sorted(set(alldat['SET']))
	
	setgroups = []
	setgroup  = []
	
	settime = []
	for i, st in enumerate(setlist):
		st0 = alldat[ alldat['SET'] == st].dt.values[0]
		stf = alldat[ alldat['SET'] == st].dt.values[-1]
		
		if i > 0:
			setdur = (st0-st1f)/np.timedelta64(1, 'm')
			if setdur > 20.*60.: # 2 hours
				setgroups.append(setgroup)
				setgroup = [st]
			else:
				setgroup.append(st)
		else:
			setgroup = [st]
			setdur = 0.
		
		t = [st,st0,stf, setdur]
		settime.append(t)
		st1f = stf
		
	setgroups.append(setgroup)
	
	# read hyperparamters
	resflines = open(resf, "r").readlines()
	for rline in resflines:
		if "ABIC" in rline:
			abic = rline.strip().split("=")[1].split()[0]
			abic = float(abic)
		if "lambda_0" in rline:
			lamb0 = rline.strip().split("=")[-1]
			lamb0 = math.log10(float(lamb0))
		if "lambda_g" in rline:
			lambg = rline.strip().split("=")[-1]
			lambg = math.log10(float(lambg))
		if "mu_t" in rline:
			mu_t = rline.strip().split("=")[-1].split()[0]
			mu_t = float(mu_t)
		if "mu_MT" in rline:
			mu_m = rline.strip().split("=")[-1].strip()
			mu_m = float(mu_m)
		if "Site_name" in rline:
			site = rline.strip().split("=")[-1].strip()
		if "Campaign" in rline:
			camp = rline.strip().split("=")[-1].strip()
	sitecamp = site + "." + camp
	
	print(sitecamp)
	
	alldat['zeros'] = 0.
	nsetg = len(setgroups)
	for isetg, setgroup in enumerate(setgroups):
		print(setgroup)
		shots = alldat[ alldat['SET'].isin(setgroup) ]
		
		x0 = d0 + timedelta(seconds=shots.ST.values[0])  - timedelta(seconds=offsec)
		x1 = d0 + timedelta(seconds=shots.ST.values[-1]) + timedelta(seconds=offsec)
		mp = np.loadtxt(mpfile, delimiter=',')
		
		if nsetg == 1:
			obsimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","t.s."+ext))
		else:
			obsimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","t.s%1d.%s"%(isetg,ext)))
		
		fig = plt.figure(figsize=(14,18))
		plt.rcParams["font.size"] = 18
		ax0 = fig.add_subplot(4, 1, 1)
		ax1 = fig.add_subplot(4, 1, 2, sharex=ax0)
		ax2 = fig.add_subplot(4, 1, 3, sharex=ax0)
		ax3 = fig.add_subplot(4, 1, 4)
		ax0.set_position([0.125, 0.7, 0.75, 0.2])
		ax1.set_position([0.125, 0.6, 0.75, 0.1])
		ax2.set_position([0.125, 0.4, 0.75, 0.2])
		ax3.set_position([0.125, 0.1, 0.75, 0.3])
		
		## Residuals ##
		shot_tmp = shots[~shots['flag']].reset_index(drop=True).copy()
		rejected = shots[ shots['flag']].reset_index(drop=True).copy()
		shot_tmp['ResiTT'] = shot_tmp['ResiTT']*1000.
		rms = lambda d: np.sqrt((d ** 2.).sum() / d.size)
		rmstt = rms( shot_tmp['ResiTT'].values )
		ir = len(rejected.index)
		nshot = len(shots.index)
		ushot = len(shot_tmp.index)
		rj = float(ir)/float(nshot)*100.
		
		y0 = max( abs(shot_tmp.ResiTT.max()), abs(shot_tmp.ResiTT.min()) )
		y0 = max(float(math.ceil(y0*10.))/10., 0.3)
		ax0.set_xlim( x0, x1)
		ax0.set_ylim(-y0, y0)
		ax0.set_ylabel(r"Travel-time redidual [ms]",fontsize=20)
		
		ax1.set_xlim( x0, x1)
		ax1.set_ylim( 0, len(MTs)+1)
		ax1.set_ylabel("Rejected\nshots\n",fontsize=20)
		ax1.tick_params(labelbottom=False)
		
		if (x1-x0) > timedelta(hours=17.):
			inthour = 8
			subhour = 4
		elif (x1-x0) > timedelta(hours=9.):
			inthour = 4
			subhour = 2
		elif (x1-x0) > timedelta(hours=5.):
			inthour = 2
			subhour = 1
		else:
			inthour = 1
			subhour = 1
		
		for i in range(len(MTs)):
			mt = MTs[i]
			xx = shot_tmp.loc[(shot_tmp['MT']==mt),'dt']
			yy = shot_tmp.loc[(shot_tmp['MT']==mt),'ResiTT']
			ax0.plot(xx, yy, marker=".", color=mtcmap(i), linestyle='None', label=mt)
			
			xr = rejected.loc[(rejected['MT']==mt),'dt']
			yr = [len(MTs)-i for x in xr]
			ax1.plot(xr, yr, marker="x", color=mtcmap(i), linestyle='None', label=mt)
		
		ax0.plot(rejected['dt'], rejected['zeros'],  marker="x", color="black", linestyle='None', label=r"Rjc." )
		ax0.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', borderaxespad=0)
		shot_tmp['a'] = -y0 + 0.01
		sets = shots['SET'].unique()
		
		vecindex = []
		for s, st in enumerate(sets):
			tm = shot_tmp[(shot_tmp['SET']==st)].reset_index(drop=True).copy()
			tw0 = tm['ST'].values[0]
			tw1 = tm['RT'].values[-1]
			s0index = shot_tmp[(shot_tmp['SET']==st)].index.values[0]
			s1index = shot_tmp[(shot_tmp['SET']==st)].index.values[-1]
			vecindex.append([st, np.linspace(s0index, s1index, ngrad)])
			ax0.plot(tm['dt'], tm['a'], color=setcmap(s), linewidth=4, zorder=1)
			ax3.plot(tm['dt'], tm['zeros'], color=setcmap(s), linewidth=4, zorder=1,label=st)
		
		ax0.xaxis.set_major_locator(mdates.HourLocator(interval=inthour, tz=None))
		ax0.xaxis.set_minor_locator(mdates.HourLocator(interval=subhour, tz=None))
		ax0.xaxis.set_major_formatter(mdates.DateFormatter("'%y-%m-%d\n%H:%M"))
		ax0.tick_params(labelbottom=False)
		ax1.tick_params(labelleft=False)
		
		## dV ##
		y0 = shot_tmp['dV'].min()-0.02
		y1 = shot_tmp['dV'].max()+0.02
		y0 = min(float(math.ceil(y0*10.))/10., float(math.floor(y0*10.))/10.)
		y1 = max(float(math.ceil(y1*10.))/10., float(math.floor(y1*10.))/10.)
		
		ax2.set_xlim(x0, x1)
		ax2.set_ylim(y0, y1)
		
		ax2.set_ylabel(r"$\delta V$ [m/s]",fontsize=20)
		aveV = r"$\overline{V_0} = $ %7.2f m/s"  % V0
		
		for i in range(len(MTs)):
			mt = MTs[i]
			xx = shot_tmp.loc[(shot_tmp['MT']==mt),'dt']
			yy = shot_tmp.loc[(shot_tmp['MT']==mt),'dV']
			ll = r"%s" % mt
			ax2.plot(xx, yy, marker="+", color=mtcmap(i), linestyle='None', label=ll)
		
		for s, st in enumerate(sets):
			
			if s == 0:
				lb0 = r"$\delta V_0$"
			else:
				lb0 = ""
			
			tm = shot_tmp[(shot_tmp['SET']==st)].reset_index(drop=True).copy()
			ax2.plot(tm.dt, tm.dV0, label=lb0, color='black')
			
		ax2.xaxis.set_major_locator(mdates.HourLocator(interval=inthour, tz=None))
		ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=subhour, tz=None))
		ax2.xaxis.set_major_formatter(mdates.DateFormatter("'%y-%m-%d\n%H:%M"))
		ax2.tick_params(labelbottom=False)
		ax2.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0)
		ax2.text(x1, y0, aveV, fontsize=16, ha="right", va="bottom")
		
		## grad SV ##
		
		offset1 =  1.0
		offset2 = -1.0
		
		gradV1 = [ (e**2.+n**2.)**0.5 for e, n in zip(shot_tmp.gradV1e,shot_tmp.gradV1n)]
		gradV2 = [ (e**2.+n**2.)**0.5 for e, n in zip(shot_tmp.gradV2e,shot_tmp.gradV2n)]
		
		scaleVs = float(math.ceil(max(gradV1)*10.))/10.
		scaleVd = float(math.ceil(max(gradV2)*10.))/10.
		scale = max(scaleVs, scaleVd)
		
		ax3.set_xlim(x0, x1)
		ax3.set_ylim(2*offset2, 2*offset1)
		ax3.set_ylabel(r"Gradient Sound Speed",fontsize=20)
		ax3.tick_params(labelleft=False)
		
		for s, st in enumerate(sets):
			
			if s == 0:
				lb1e = r"$g_1^E$"
				lb1n = r"$g_1^N$"
				lb2e = r"$g_2^E$"
				lb2n = r"$g_2^N$"
			else:
				lb1e = ""
				lb1n = ""
				lb2e = ""
				lb2n = ""
			
			tm = shot_tmp[(shot_tmp['SET']==st)].reset_index(drop=True).copy()
			ax3.plot(tm.dt, tm.gradV1e/scale+offset1, label=lb1e, color='mediumblue', linestyle='dashed')
			ax3.plot(tm.dt, tm.gradV1n/scale+offset1, label=lb1n, color='mediumblue', linestyle='solid' )
			ax3.plot(tm.dt, tm.gradV2e/scale+offset2, label=lb2e, color='navy', linestyle='dashed')
			ax3.plot(tm.dt, tm.gradV2n/scale+offset2, label=lb2n, color='navy', linestyle='solid' )
			
			idxnp =  vecindex[s][1]
			
			xx  = [shot_tmp.loc[int(idx),"dt"] for idx in idxnp]
			yy1 = [offset1 for idx in idxnp]
			yy2 = [offset2 for idx in idxnp]
			es  = [shot_tmp.loc[int(idx),"gradV1e"] for idx in idxnp]
			ns  = [shot_tmp.loc[int(idx),"gradV1n"] for idx in idxnp]
			ed  = [shot_tmp.loc[int(idx),"gradV2e"] for idx in idxnp]
			nd  = [shot_tmp.loc[int(idx),"gradV2n"] for idx in idxnp]
			
			wd = 0.003
			q1 = ax3.quiver(xx, yy1, es, ns, width=wd, label=None, color='mediumblue', zorder=2, scale_units='y', scale=scale)
			q2 = ax3.quiver(xx, yy2, ed, nd, width=wd, label=None, color='darkviolet', zorder=2, scale_units='y', scale=scale)
			
			if s == 0:
				ax3.quiverkey(q1, 1.08, 0.9, scale, r'$\bf g_1$'+'\n%4.2f m/s/km'%scale, labelpos='N', color="mediumblue")#, coordinates='figure')
				ax3.quiverkey(q2, 1.08, 0.1, scale, r'$\bf g_2$'+'\n%4.2f m/s/km'%scale, labelpos='S', color="darkviolet")#, coordinates='figure')
		
		ax3.xaxis.set_major_locator(mdates.HourLocator(interval=inthour, tz=None))
		ax3.xaxis.set_minor_locator(mdates.HourLocator(interval=subhour, tz=None))
		ax3.xaxis.set_major_formatter(mdates.DateFormatter("'%y-%m-%d\n%H:%M"))
		ax3.legend(bbox_to_anchor=(1.03, 0.5), loc='center left', borderaxespad=0)
		ax3.set_xlabel(r"Time (UTC)", fontsize=20)
		
		suptl = sitecamp
		
		supt  = "RMS(Residual) = %6.4f ms : " % rmstt
		supt += "%4d/%5d-shot (%4.1f%%) rejected\n" % (ir,nshot,rj)
		supt += r'ABIC = %12.4f, ' % abic
		supt += r'$\lambda_0^2 = 10^{%4.1f}$, ' % lamb0
		supt += r'$\lambda_g^2 = 10^{%4.1f}$, ' % lambg
		supt += r'$\mu_t = %4.1f$ min., ' % (mu_t/60.)
		supt += r'$\mu_{MT} = %4.2f$' % mu_m
		
		
		fig.suptitle(suptl, fontsize = 30, x=0.5)
		ax0.set_title(supt, fontsize = 22, loc="right")
		
		ax0.grid(which = "minor", axis = "x", linestyle = "--", linewidth = 1)
		ax1.grid(which = "minor", axis = "x", linestyle = "--", linewidth = 1)
		ax2.grid(which = "minor", axis = "x", linestyle = "--", linewidth = 1)
		ax3.grid(which = "minor", axis = "x", linestyle = "--", linewidth = 1)
		ax0.grid(which = "major", axis = "y", linestyle = "--", linewidth = 1)
		ax2.grid(which = "major", axis = "y", linestyle = "--", linewidth = 1)
		ax3.grid(which = "major", axis = "y", linestyle = "--", linewidth = 1)
		plt.savefig(obsimg)
		plt.close()
	
	return
	

if __name__ == '__main__':
	######################################################################
	usa = u"Usage: %prog [options] "
	opt = OptionParser(usage=usa)
	opt.add_option( "--resfiles", action="store", type="string",
					default="", dest="resfls",
					help=u"Set result file names"
					)
	opt.add_option( "--ext", action="store", type="string",
					default="png", dest="ext",
					help=u"Set extention of output (default: png)"
					)
	(options, args) = opt.parse_args()
	#####################################################################
	
	resfs = options.resfls.strip('"').strip("'")
	resfiles = glob.glob(resfs)
	resfiles.sort()
	
	if len(resfiles) == 0:
		print("NOT FOUND (res.dat files) :: %s" % options.resfls)
		sys.exit(1)
	
	for resf in resfiles:
		
		cfg = configparser.ConfigParser()
		cfg.read(resf, 'UTF-8')
		
		### Check site-name ###
		site  = cfg.get("Obs-parameter", "Site_name")
		date1 = cfg.get("Obs-parameter", "Date(UTC)")
		MTs  = cfg.get("Site-parameter", "Stations").split(" ")
		MTs = [ str(mt) for mt in MTs]
		
		### Sound speed profile ###
		# calc depth-averaged sound speed (characteristic length/time)
		svpf = cfg.get("Obs-parameter", "SoundSpeed")
		svp = pd.read_csv(svpf, comment='#')
		vl = svp.speed.values
		dl = svp.depth.values
		avevlyr = [ (vl[i+1]+vl[i])*(dl[i+1]-dl[i])/2. for i in svp.index[:-1]]
		V0 = np.array(avevlyr).sum()/(dl[-1]-dl[0])
		
		# plot obsfile
		obsf = cfg.get("Data-file", "datacsv")
		mpf  = resf.replace('res.dat','m.p.dat')
		d0 = datetime.datetime.strptime(date1,"%Y-%m-%d")
		
		plot_residuals(resf, obsf, mpf, d0, MTs, V0, options.ext)
		
	
	exit()
	
