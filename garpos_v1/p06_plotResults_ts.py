#!/usr/bin/python3
# -*- coding: utf-8 -*-
############################################

from scipy import fftpack
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.patches as patches
#from astropy.timeseries import LombScargle as LS
from scipy.signal import lombscargle as LS
import pandas as pd
import matplotlib.dates as mdates
import math, datetime, os
from scipy.special import chebyt
from pandas import plotting 
import seaborn as sns
from module.utilities import *

"""
class Spectra(object):
	def __init__(self, t, f, time_unit):
		# - t : 時間軸の値
		# - f : データの値
		# - time_unit : 時間軸の単位
		# - Po : パワースペクトル密度
		assert t.size == f.size  # 時間軸の長さとデータの長さが同じであることを確認する
		#assert np.unique(np.diff(t)).size == 1  # 時間間隔が全て一定であることを確認する
		assert max(np.unique(np.diff(t))) - min(np.unique(np.diff(t))) <= 1.e-8  # 時間間隔が全て一定であることを確認する
		self.time_unit = time_unit   # 時間の単位
		T = (t[1] - t[0]) * t.size
		self.period = 1.0 / (np.arange(t.size / 2)[1:] / T)
		
		# パワースペクトル密度を計算
		f = f - np.average(f)         # 平均をゼロに。
		F = fftpack.fft(f)                          # 高速フーリエ変換
		self.Po = np.abs(F[1:(t.size // 2)]) ** 2 / T
	
	def draw_with_time(self, fsizex=8, fsizey=6, print_flg=True, threshold=1.0):
		# 横軸に時間をとってパワースペクトル密度をプロット
		fig, ax = plt.subplots(figsize=(fsizex, fsizey))   # 図のサイズの指定
		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.set_xlabel(self.time_unit)
		ax.set_ylabel("Power Spectrum Density")
		ax.plot(self.period, self.Po)
		if print_flg:   # パワースペクトル密度の値がthresholdより大きい部分に線を引き、周期の値を記述する
			dominant_periods = self.period[self.Po > threshold]
			print(dominant_periods, self.time_unit +
				  ' components are dominant!')
			for dominant_period in dominant_periods:
				plt.axvline(x=dominant_period, linewidth=0.5, color='k')
				ax.text(dominant_period, threshold,
						str(round(dominant_period, 3)))
		
		return plt
"""
def make_outlier_criteria(col):
	outlier_min =  2. #average - (sd) * 1
	outlier_max = -2. #average + (sd) * 1
	return outlier_max,outlier_min

def drop_outlier(df_new):
#Cut DataFrame
	df_without_outlier = pd.DataFrame([])
#Remove Outlier
	df_temp = df_new
#	out_e = make_outlier_criteria(df_temp['Vs'].values)
	df_temp[df_temp['Vd_Vs'] >= 1.0] = 1000.
	df_temp[df_temp['Vd_Vs'] <  0.0] = 1000.
#	df_temp[df_temp['Vs_e'] <= out_e[0]] = None
#	df_temp[df_temp['Vs_e'] >= out_e[1]] = None
#	df_temp[df_temp['Vs_n'] <= out_n[0]] = None
#	df_temp[df_temp['Vs_n'] >= out_n[1]] = None
	df_without_outlier = df_without_outlier.append(df_temp)
#Compile DataFrame
	df_without_outlier = df_without_outlier.dropna()
	return df_without_outlier

def cov2partialcorr(cov):
	#逆行列を求める
	omega=np.linalg.inv(cov)
	D=np.diag(np.power(np.diag(omega),-0.5))
	partialcorr=-np.dot(np.dot(D,omega),D)
	#対角成分を-1から1にする
	partialcorr+=2*np.eye(cov.shape[0])
	return partialcorr

def generate_cmap(colors):
	import numpy as np
	from matplotlib.colors import LinearSegmentedColormap
	"""自分で定義したカラーマップを返す"""
	values = range(len(colors))
	
	vmax = np.ceil(np.max(values))
	color_list = []
	for v, c in zip(values, colors):
		color_list.append( ( v/ vmax, c) )
	return LinearSegmentedColormap.from_list('custom_cmap', color_list)
	
def plot_corMap(corfile, covfile, mpnum, nset):
	corimg = corfile.replace(".dat",".png")
	pcorimg = corfile.replace("cor.dat","pcor.png")
	cor = np.loadtxt(corfile, delimiter=',')
	cov = np.loadtxt(covfile, delimiter=',')
	
	mp0 = [0]
	mp1 = [0]
	sta = []
	s0  = []
	s1  = []
	for i, mn in enumerate(mpnum):
		mp0.append(mp0[i]+mn)
		if i >= 1:
			for j in range(nset-1):
				if mn % nset != 0:
					print("error")
					exit()
				ss = int(mn/nset)
				s0.append(ss*(j+1) + mp0[i])
				if j == 0:
					s1.append(ss*(j+1) + mp0[i])
					mp1.append(mp1[-1] + ss)
		else:
			s1.append(mn)
			mp1.append(mn)
			ista = int(mn/3)
			for ist in range(ista):
				if (ist+1)*3 != mn:
					sta.append((ist+1)*3)
	#print(mp1,mp0,sta)
	oline = [0,mp0[-1]]
	
	#colormap = plt.cm.RdBu_r
	#colormap = generate_cmap(['navy','darkblue', 'royalblue', 'cornflowerblue', '#EEEEEE', '#EEEEEE', '#EEEEEE', 'coral', 'salmon', 'orangered', 'red'])
	colormap = generate_cmap(['navy', 'royalblue', 'cornflowerblue', '#EEEEEE', '#EEEEEE', '#EEEEEE', 'coral', 'salmon', 'orangered'])
	
	fig = plt.figure(figsize=(14,12))
	plt.title('Correlation Matrix for '+ os.path.basename(corfile), y=1.05, size=15)
	ax = sns.heatmap(cor,linewidths=0.1,vmax=1.0, vmin=-1.0, xticklabels=False, yticklabels=False,
					 square=True, cmap=colormap, linecolor='white', annot=False)
	
	for m in mp0[1:-1]:
		ax.plot(oline, [m,m], 'g-', lw = 1)
		ax.plot([m,m], oline, 'g-', lw = 1)
	for s in sta:
		ax.plot(oline, [s,s], 'y-', lw = 0.5)
		ax.plot([s,s], oline, 'y-', lw = 0.5)
	for s in s0:
		ax.plot(oline, [s,s], 'y-', lw = 0.5)
		ax.plot([s,s], oline, 'y-', lw = 0.5)
	
	#plt.savefig(corimg)
	plt.close()
	
	#Partial correlation
	pcor = cov2partialcorr(cov)
	fig = plt.figure(figsize=(14,12))
	plt.title('Partial-Correlation Matrix for '+ os.path.basename(corfile), y=1.05, size=15)
	ax = sns.heatmap(pcor,linewidths=0.1,vmax=0.5, vmin=-0.5, xticklabels=False, yticklabels=False,
					 square=True, cmap=colormap, linecolor='white', annot=False)
	
	for m in mp0[1:-1]:
		ax.plot(oline, [m,m], 'g-', lw = 1)
		ax.plot([m,m], oline, 'g-', lw = 1)
	for s in sta:
		ax.plot(oline, [s,s], 'y-', lw = 0.5)
		ax.plot([s,s], oline, 'y-', lw = 0.5)
	for s in s0:
		ax.plot(oline, [s,s], 'y-', lw = 0.5)
		ax.plot([s,s], oline, 'y-', lw = 0.5)
	
	plt.savefig(pcorimg)
	plt.close()
	
	"""
	####################
	# 1set corr matrix #
	####################
	corimg = corfile.replace(".dat","1set.png")
	pcorimg = corfile.replace("cor.dat","pcor1set.png")
	
	i1set = []
	for i in range(len(s1)):
		i1set += list(range(mp0[i],s1[i]))
	
	cor = cor[np.ix_(i1set,i1set)]
	pcor = pcor[np.ix_(i1set,i1set)]
	
	fig = plt.figure(figsize=(14,12))
	plt.title('1-set Correlation Matrix for '+ os.path.basename(corfile), y=1.05, size=15)
	ax = sns.heatmap(cor,linewidths=0.1,vmax=1.0, vmin=-1.0, xticklabels=False, yticklabels=False,
					 square=True, cmap=colormap, linecolor='white', annot=False)
	
	for m in mp1[1:-1]:
		ax.plot(oline, [m,m], 'k-', lw = 1)
		ax.plot([m,m], oline, 'k-', lw = 1)
	for s in sta:
		ax.plot(oline, [s,s], 'g-', lw = 0.5)
		ax.plot([s,s], oline, 'g-', lw = 0.5)
	
	plt.savefig(corimg)
	plt.close()
	
	#Partial correlation
	#pcor = cov2partialcorr(cov)
	fig = plt.figure(figsize=(14,12))
	plt.title('1-set Partial-Correlation Matrix for '+ os.path.basename(corfile), y=1.05, size=15)
	ax = sns.heatmap(pcor,linewidths=0.1,vmax=0.5, vmin=-0.5, xticklabels=False, yticklabels=False,
					 square=True, cmap=colormap, linecolor='white', annot=False)
	
	for m in mp1[1:-1]:
		ax.plot(oline, [m,m], 'k-', lw = 1)
		ax.plot([m,m], oline, 'k-', lw = 1)
	for s in sta:
		ax.plot(oline, [s,s], 'g-', lw = 0.5)
		ax.plot([s,s], oline, 'g-', lw = 0.5)
	
	plt.savefig(pcorimg)
	plt.close()
	#print(pcor[2,mp1[1]])
	"""
	return
	
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
	if x.size != y.size:
		raise ValueError("x and y must be the same size")

	cov = np.cov(x, y)
	pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
	ell_radius_x = np.sqrt(1 + pearson)
	ell_radius_y = np.sqrt(1 - pearson)
	ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
	scale_x = np.sqrt(cov[0, 0]) * n_std
	mean_x = np.mean(x)

	scale_y = np.sqrt(cov[1, 1]) * n_std
	mean_y = np.mean(y)

	transf = transforms.Affine2D() \
					.rotate_deg(45) \
					.scale(scale_x, scale_y) \
					.translate(mean_x, mean_y)

	ellipse.set_transform(transf + ax.transData)
	return ax.add_patch(ellipse)

def plot_residuals(resf, obsfile, mpfile, d0, MTs, scalel):
	
	figdir = os.path.dirname(resf)+"/fig"
	
	mkdirectory(figdir)
	
	offsec = 60.*60.
	ngrad  = 6
	setcmap = plt.get_cmap("Set3")
	mtcmap  = plt.get_cmap("tab10")
	
	all_shot_dat = pd.read_csv(obsfile, comment='#', index_col=0)
	all_shot_dat['dt'] = [ d0 + datetime.timedelta(seconds=t) for t in all_shot_dat['ST'] ]
	
	setlist = sorted(set(all_shot_dat['SET']))
	
	setgroups = []
	setgroup  = []
	
	settime = []
	for i, st in enumerate(setlist):
		st0 = all_shot_dat[ all_shot_dat['SET'] == st].dt.values[0]
		stf = all_shot_dat[ all_shot_dat['SET'] == st].dt.values[-1]
		
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
	
	all_shot_dat['zeros'] = 0.
	nsetg = len(setgroups)
	for isetg, setgroup in enumerate(setgroups):
		print(setgroup)
		shot_dat = all_shot_dat[ all_shot_dat['SET'].isin(setgroup) ]
		
		x0 = d0 + datetime.timedelta(seconds=shot_dat['ST'].values[0])  - datetime.timedelta(seconds=offsec)
		x1 = d0 + datetime.timedelta(seconds=shot_dat['ST'].values[-1]) + datetime.timedelta(seconds=offsec)
		mp = np.loadtxt(mpfile, delimiter=',')
		
		if nsetg == 1:
			#acfimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","acf.png"))
			obsimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","t.s.png"))
		else:
			#acfimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","acf%1d.png"%(isetg)))
			obsimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","t.s%1d.png"%(isetg)))
		
		fig = plt.figure(figsize=(14,18))
		plt.rcParams["font.size"] = 14
		ax0 = fig.add_subplot(4, 1, 1) 
		ax1 = fig.add_subplot(4, 1, 2, sharex=ax0) 
		ax2 = fig.add_subplot(4, 1, 3, sharex=ax0) 
		ax3 = fig.add_subplot(4, 1, 4) 
		ax0.set_position([0.125,0.7,0.75,0.2])
		ax1.set_position([0.125,0.6,0.75,0.1])
		ax2.set_position([0.125,0.4,0.75,0.2])
		ax3.set_position([0.125,0.1,0.75,0.3])
		
		## Residuals ##
		shot_tmp = shot_dat[~shot_dat['flag']].reset_index(drop=True).copy()
		rejected = shot_dat[ shot_dat['flag']].reset_index(drop=True).copy()
		shot_tmp['ResiTT'] = shot_tmp['ResiTT']*1000.
		rms = lambda d: np.sqrt((d ** 2.).sum() / d.size)
		rmstt = rms( shot_tmp['ResiTT'].values )
		ir = len(rejected.index)
		nshot = len(shot_dat.index)
		ushot = len(shot_tmp.index)
		rj = float(ir)/float(nshot)*100.
		
		y0 = max( abs(shot_tmp['ResiTT'].max()), abs(shot_tmp['ResiTT'].min()) )
		y0 = max(float(math.ceil(y0*10.))/10., 0.3)
		ax0.set_xlim( x0, x1)
		ax0.set_ylim(-y0, y0)
		ax0.set_ylabel("T.T. redidual [ms]",fontsize=16)
		
		ax1.set_xlim( x0, x1)
		ax1.set_ylim( 0, len(MTs)+1)
		ax1.set_ylabel("Rejected\nshots\n",fontsize=16)
		ax1.tick_params(labelbottom=False)
		
		for i in range(len(MTs)):
			mt = MTs[i]
			xx = shot_tmp.loc[(shot_tmp['MT']==mt),'dt']
			yy = shot_tmp.loc[(shot_tmp['MT']==mt),'ResiTT']
			ax0.plot(xx, yy, marker=".", color=mtcmap(i), linestyle='None', label=mt)
			
			xr = rejected.loc[(rejected['MT']==mt),'dt']
			yr = [len(MTs)-i for x in xr]
			ax1.plot(xr, yr, marker="x", color=mtcmap(i), linestyle='None', label=mt)
		
		ax0.plot(rejected['dt'], rejected['zeros'],  marker="x", color="black", linestyle='None', label="Rjc." )
		ax0.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', borderaxespad=0)
		shot_tmp['a'] = -y0 + 0.01
		sets = shot_dat['SET'].unique()
		
		vecindex = []
		for s, st in enumerate(sets):
			tm = shot_tmp[(shot_tmp['SET']==st)].reset_index(drop=True).copy()
			tw0 = tm['ST'].values[0]
			tw1 = tm['RT'].values[-1]
			s0index = shot_tmp[(shot_tmp['SET']==st)].index.values[0]
			s1index = shot_tmp[(shot_tmp['SET']==st)].index.values[-1]
			vecindex.append([st, np.linspace(s0index, s1index, ngrad)])
			ax0.plot(tm['dt'], tm['a'], color=setcmap(s), linewidth=4, zorder=1)
			ax3.plot(tm['ST'], tm['zeros'], color=setcmap(s), linewidth=4, zorder=1,label=st)
		
		ax0.xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=None))
		ax0.xaxis.set_major_formatter(mdates.DateFormatter("'%y-%m-%d\n%H:%M"))
		ax1.tick_params(labelleft=False)
		ax0.xaxis.tick_top()
		
		## dSV ##
		y0 = shot_tmp['dV0'].min()
		y1 = shot_tmp['dV0'].max()
		y0 = min(float(math.ceil(y0*10.))/10., float(math.floor(y0*10.))/10.)
		y1 = max(float(math.ceil(y1*10.))/10., float(math.floor(y1*10.))/10.)
		
		ax2.set_xlim(x0, x1)
		ax2.set_ylim(y0, y1)
####
#		rttf = resf.replace('res.dat','rtt.dat')
#		rtt = pd.read_csv(filepath_or_buffer=rttf, delim_whitespace=True)
#		ax2.set_ylabel("%7.2f + d-Sound Speed [m/s]" % (rtt.avedV2.values[-1]),fontsize=16)
		ax2.set_ylabel("d-Sound Speed [m/s]",fontsize=16)
####
		for i in range(len(MTs)):
			mt = MTs[i]
			xx = shot_tmp.loc[(shot_tmp['MT']==mt),'dt']
			yy = shot_tmp.loc[(shot_tmp['MT']==mt),'dV']
			ax2.plot(xx, yy, marker="+", color=mtcmap(i), linestyle='None')
		
		ax2.plot(shot_tmp.dt, shot_tmp.dV0, color='black')
#		ax2.plot(shot_tmp.dt, shot_tmp.dVbulk0, color='black')
		ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=None))
		ax2.xaxis.set_major_formatter(mdates.DateFormatter("'%y-%m-%d\n%H:%M"))
		ax2.tick_params(labelbottom=False)
		
		## grad SV ##
		
		offset1 =  1.0
		offset2 = -1.0
		
#gradVs-gradVdの処理は手前で
		shot_tmp.gradV1e=shot_tmp.gradV1e+shot_tmp.gradV2e
		shot_tmp.gradV1n=shot_tmp.gradV1n+shot_tmp.gradV2n
		gradVs = [ (e**2.+n**2.)**0.5 for e, n in zip(shot_tmp.gradV1e,shot_tmp.gradV1n)]
		gradVd = [ (e**2.+n**2.)**0.5 for e, n in zip(shot_tmp.gradV2e,shot_tmp.gradV2n)]
		
		scaleVs = float(math.ceil(max(gradVs)*10.))/10.
		scaleVd = float(math.ceil(max(gradVd)*10.))/10.
		scale = max(scaleVs, scaleVd)
		#offset1 = scale
		#if nmpsv2 != 0:
		#	offset2 = float(math.ceil(max(gradVd)*10.))/10.
		#else:
		#	offset2 = 1.
		#offset2 = -scale
		
		ax3.set_xlim(shot_dat['ST'].values[0]-offsec, shot_dat['ST'].values[-1]+offsec)
		ax3.set_ylim(4*offset2+2*offset2, 2*offset1)
		#ax3.set_ylabel("grad-Sound Speed [m/s/km]\n(%3.1f m/s/km for 1-ytic)\n"%offset,fontsize=16)
		ax3.set_ylabel("grad-Sound Speed",fontsize=16)
		ax3.tick_params(labelleft=False)
		
		ax3.plot(shot_tmp.ST, shot_tmp.gradV1e/scale+offset1, color='mediumblue', linestyle='dashed', label='g1:E')
		ax3.plot(shot_tmp.ST, shot_tmp.gradV1n/scale+offset1, color='mediumblue', linestyle='solid' , label='g1:N')
		
		ax3.plot(shot_tmp.ST, shot_tmp.gradV2e/scale+offset2, color='darkviolet', linestyle='dashed', label='g2:E')
		ax3.plot(shot_tmp.ST, shot_tmp.gradV2n/scale+offset2, color='darkviolet', linestyle='solid',  label='g2:N')
####
		## g2 + g1 * i
#		gradVe = [ (s**2.+d**2.)**0.5 for s, d in zip(shot_tmp.gradV1e,shot_tmp.gradV2e)]
#		gradVn = [ (s**2.+d**2.)**0.5 for s, d in zip(shot_tmp.gradV1n,shot_tmp.gradV2n)]
#		radVe  = [ math.atan2(d,s)    for s, d in zip(shot_tmp.gradV1e,shot_tmp.gradV2e)]
#		radVn  = [ math.atan2(d,s)    for s, d in zip(shot_tmp.gradV1n,shot_tmp.gradV2n)]
		gradVe = [ (s**2.+d**2.)**0.5 for s, d in zip(shot_tmp.gradV1e,shot_tmp.gradV2e)]
		gradVn = [ (s**2.+d**2.)**0.5 for s, d in zip(shot_tmp.gradV1n,shot_tmp.gradV2n)]
		radVe  = [ math.atan2(s,d)    for s, d in zip(shot_tmp.gradV1e,shot_tmp.gradV2e)]
		radVn  = [ math.atan2(s,d)    for s, d in zip(shot_tmp.gradV1n,shot_tmp.gradV2n)]
		shot_tmp = shot_tmp.assign(A_Vd_iVs_E=gradVe, A_Vd_iVs_N=gradVn, Vd_iVs_E=radVe, Vd_iVs_N=radVn)

		scale_gradVe = float(math.ceil(max(gradVe)*10.))/10.
		scale_gradVn = float(math.ceil(max(gradVn)*10.))/10.
		scale_grad = max(scale_gradVe,scale_gradVn)
		
		x1 = np.arange(shot_dat['ST'].values[0]-offsec, shot_dat['ST'].values[-1]+offsec)
		formul =["$g_{2}+ig_{1}$ E:%7.2f N:%7.2f [cm/s/km]"%(max(gradVe)*100.,max(gradVn)*100.),"$\pi/2$","$\pi/3$","$-\pi/2$","$-2\pi/3$"]
#		formul =["$g_{2}+ig_{1}$","$\pi/2$","$\pi/4$","$-\pi/2$","$-3\pi/4$"]
		offset3=[2.000*offset2,2.500*offset2,2.667*offset2,3.500*offset2,3.667*offset2]
#		offset3=[2.000*offset2,2.500*offset2,2.750*offset2,3.500*offset2,3.750*offset2]
		ax3.text(shot_dat['ST'].values[0]-offsec+100,offset3[0],formul[0],fontsize=10)
		for i in range(4):
			ax3.text(shot_dat['ST'].values[0]-offsec+100,offset3[i+1],formul[i+1],fontsize=8)
		for i in range(4):
			ax3.text(shot_dat['ST'].values[0]-offsec+100,2.0*offset2+offset3[i+1],formul[i+1],fontsize=8)
		formul =["East","West","North","South"]
		offset3=[2.666*offset2,3.666*offset2,4.666*offset2,5.666*offset2]
		for i in range(4):
			ax3.text(shot_dat['ST'].values[-1],offset3[i],formul[i],fontsize=11)
		offset3=[1.000,0.500,0.333,0.000,-0.500,-0.666]
#		offset3=[1.000,0.500,0.250,0.000,-0.500,-0.750]
		for i in range(6):
			ax3.plot(x1, offset3[i]+x1*0+3.0*offset2, color='gray', linestyle='dashed', linewidth=0.5 )
		for i in range(5):
			ax3.plot(x1, offset3[i+1]+x1*0+5.0*offset2, color='gray', linestyle='dashed', linewidth=0.5 )
		ax3.plot(x1, -1.000+x1*0+3.0*offset2, color='black', linestyle='solid' )
		colorsE = cm.Blues(shot_tmp.A_Vd_iVs_E/scale_grad)
		colorsN = cm.Greens(shot_tmp.A_Vd_iVs_N/scale_grad)
		formul = ['#FF46FF','#FF6EFF','#FF96FF','#FFBEFF',  '#FFFF46','#FFFF6E','#FFFF96','#FFFFBE',  '#F71919','#F74A4A','#F77C7C','#F7ADAD',  '#FFAA19','#FFC259','#FFD999','#FFF0DB']
		offset3=[0.50+3.0*offset2+0.50*0.8,  0.50+3.0*offset2+0.50*0.6,  0.50+3.0*offset2+0.50*0.4, 0.50+3.0*offset2+0.50*0.2,  -0.50+3.0*offset2+0.50*0.8,-0.50+3.0*offset2+0.50*0.6,-0.50+3.0*offset2+0.50*0.4,-0.50+3.0*offset2+0.50*0.2, 0.50+5.0*offset2+0.500*0.8, 0.50+5.0*offset2+0.500*0.6, 0.50+5.0*offset2+0.500*0.4, 0.50+5.0*offset2+0.500*0.2,  -0.50+5.0*offset2+0.500*0.8, -0.50+5.0*offset2+0.500*0.6, -0.50+5.0*offset2+0.500*0.4, -0.50+5.0*offset2+0.500*0.2]
		for s in range(16):
			r = patches.Rectangle(xy=(shot_dat['ST'].values[0]-offsec, offset3[s]), width=shot_dat['ST'].values[-1]-shot_dat['ST'].values[0]+2*offsec, height=0.500*0.2, fc=formul[s], ec='#FFFFFF')
			ax3.add_patch(r)
		formul = ['#FFFFBE','#FFFF86','#FFFF46',  '#FFBEFF','#FF86FF','#FF46FF',  '#FFE9B9','#FFC969','#FFAA19',  '#F7ACAC','#F76565','#F71919']
#		offset3=[0.00+3.0*offset2+0.166*0.8, 0.00+3.0*offset2+0.166*0.4, 0.00+3.0*offset2,  -1.00+3.0*offset2+0.166*0.8,-1.00+3.0*offset2+0.166*0.4,-1.00+3.0*offset2,  0.00+5.0*offset2+0.166*0.8, 0.00+5.0*offset2+0.166*0.4, 0.00+5.0*offset2,  -1.00+5.0*offset2+0.166*0.8, -1.00+5.0*offset2+0.166*0.4, -1.00+5.0*offset2]
		offset3=[0.00+3.0*offset2+0.250*0.8, 0.00+3.0*offset2+0.250*0.4, 0.00+3.0*offset2,  -1.00+3.0*offset2+0.250*0.8,-1.00+3.0*offset2+0.250*0.4,-1.00+3.0*offset2,  0.00+5.0*offset2+0.250*0.8, 0.00+5.0*offset2+0.250*0.4, 0.00+5.0*offset2,  -1.00+5.0*offset2+0.250*0.8, -1.00+5.0*offset2+0.250*0.4, -1.00+5.0*offset2]
		for s in range(12):
			r = patches.Rectangle(xy=(shot_dat['ST'].values[0]-offsec, offset3[s]), width=shot_dat['ST'].values[-1]-shot_dat['ST'].values[0]+2*offsec, height=0.250*0.4, fc=formul[s], ec='#FFFFFF')
#			r = patches.Rectangle(xy=(shot_dat['ST'].values[0]-offsec, offset3[s]), width=shot_dat['ST'].values[-1]-shot_dat['ST'].values[0]+2*offsec, height=0.166*0.4, fc=formul[s], ec='#FFFFFF')
			ax3.add_patch(r)

		for s in range(20,shot_tmp.ST.shape[0],20):
			ax3.plot(shot_tmp.ST[s], shot_tmp.Vd_iVs_E[s]/math.pi+3.0*offset2, color=colorsE[s], marker='s', markersize=3, linestyle='None' )
			ax3.plot(shot_tmp.ST[s], shot_tmp.Vd_iVs_N[s]/math.pi+5.0*offset2, color=colorsN[s], marker='s', markersize=3, linestyle='None' )
####
		
		for s, st in enumerate(sets):
			idxnp =  vecindex[s][1]
			
			xx  = [shot_tmp.loc[int(idx),"ST"] for idx in idxnp]
			yy1 = [offset1 for idx in idxnp]
			yy2 = [offset2 for idx in idxnp]
			es  = [shot_tmp.loc[int(idx),"gradV1e"] for idx in idxnp]
			ns  = [shot_tmp.loc[int(idx),"gradV1n"] for idx in idxnp]
			ed  = [shot_tmp.loc[int(idx),"gradV2e"] for idx in idxnp]
			nd  = [shot_tmp.loc[int(idx),"gradV2n"] for idx in idxnp]
			
#			if s == 0:
#				labels = "$g_{1}$"
#				labeld = "$g_{2}$" 
#			else:
#				labels = None
#				labeld = None
			
			qs = ax3.quiver(xx, yy1, es, ns, width=0.003,label='_nolegend_', color='mediumblue', zorder=2, scale_units='y', scale=scale)
			qd = ax3.quiver(xx, yy2, ed, nd, width=0.003,label='_nolegend_', color='darkviolet', zorder=2, scale_units='y', scale=scale)
			
			if s == 0:
				ax3.quiverkey(qs, 0.9, 0.97, scale, '%4.2f m/s/km'%scale, labelpos='S', color="black")#, coordinates='figure')
				ax3.quiverkey(qd, 0.9, 0.56, scale, '%4.2f m/s/km'%scale, labelpos='S', color="black")#, coordinates='figure')
		
		ax3.legend(bbox_to_anchor=(1.01, 0), loc='lower left', borderaxespad=0)
		fig.suptitle("%s \nRMS(Residual) = %6.4f ms : %4d/%5d-shot (%4.1f%%) rejected" % (os.path.basename(obsfile),rmstt,ir,nshot,rj), fontsize = 18)
		
		ax0.grid(which = "major", axis = "x", linestyle = "--", linewidth = 1)
		ax1.grid(which = "major", axis = "x", linestyle = "--", linewidth = 1)
		ax2.grid(which = "major", axis = "x", linestyle = "--", linewidth = 1)
		ax3.grid(which = "major", axis = "x", linestyle = "--", linewidth = 1)
		ax0.grid(which = "major", axis = "y", linestyle = "--", linewidth = 1)
		ax2.grid(which = "major", axis = "y", linestyle = "--", linewidth = 1)
		ax3.grid(which = "major", axis = "y", linestyle = "--", linewidth = 1)
		plt.savefig(obsimg)
		plt.close()
####
		obsimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","grad.png"))
		fig = plt.figure(figsize=(5,8))
		plt.rcParams["font.size"] = 6

		axE = fig.add_subplot(2,1,1) 
		axN = fig.add_subplot(2,1,2) 
		scalel=float(scalel)
		axE.set_xlim( -scalel, scalel)
		axN.set_xlim( -scalel, scalel)
		axE.set_ylim( -scalel, scalel)
		axN.set_ylim( -scalel, scalel)
		axE.set_aspect('equal')
		axN.set_aspect('equal')
		tx1 = [0,-100,-100,0]
		ty1 = [0,0,100,100]
		tx2 = [0,100,100,0]
		ty2 = [0,0,-100,-100]
		tx3 = [0,100,100]
		ty3 = [0,0,200]
		tx4 = [0,-100,-100]
		ty4 = [0,0,-200]
		axE.fill(tx1,ty1,facecolor='gray',alpha=0.5)
		axE.fill(tx2,ty2,facecolor='gray',alpha=0.5)
		axE.fill(tx3,ty3,facecolor='gainsboro',alpha=0.5)
		axE.fill(tx4,ty4,facecolor='gainsboro',alpha=0.5)
		axN.fill(tx1,ty1,facecolor='gray',alpha=0.5)
		axN.fill(tx2,ty2,facecolor='gray',alpha=0.5)
		axN.fill(tx3,ty3,facecolor='gainsboro',alpha=0.5)
		axN.fill(tx4,ty4,facecolor='gainsboro',alpha=0.5)

#		Correlation Matrix for '+ os.path.basename(corfile), y=1.05, size=15)
		for s in range(20,shot_tmp.ST.shape[0],20):
			tm = shot_tmp.SET[s]
			axE.plot(shot_tmp.gradV2e[s],shot_tmp.gradV1e[s], color=setcmap(int(shot_tmp.SET[s][1:])-1), marker='s', markersize=1, linestyle='None')
			axN.plot(shot_tmp.gradV2n[s],shot_tmp.gradV1n[s], color=setcmap(int(shot_tmp.SET[s][1:])-1), marker='s', markersize=1, linestyle='None')
			axE.text(scalel*0.165/0.20,scalel*0.18/0.20,"East")
			axN.text(scalel*0.155/0.20,scalel*0.18/0.20,"North")
			axE.text(-scalel*0.19/0.20,-scalel*0.19/0.20,"West")
			axN.text(-scalel*0.19/0.20,-scalel*0.19/0.20,"South")
		tr=0
		confidence_ellipse(shot_tmp.gradV2e, shot_tmp.gradV1e, axE, edgecolor='red')
		confidence_ellipse(shot_tmp.gradV2n, shot_tmp.gradV1n, axN, edgecolor='red')
		fig.suptitle(obsimg)
		plt.savefig(obsimg)
		plt.close()
####
		
		"""
		# calc cAIC
		pi = math.acos(-1.0)
		ndata = nshot
		nmp   = len(mp[np.where(mp[:,1]>=1.e-6),0][0])
		
		caic = ndata * (math.log(2.*pi) + 2.*math.log(rmstt) + 1.0) + float(2*nmp*ndata)/float(ndata-nmp-1)
		#print ndata, nmp, rmstt, ndata * (math.log(2.*pi) + 2.*math.log(rmstt) + 1.0) ,float(2*nmp*ndata)/float(ndata-nmp-1)
		
		# calc BIC
		pi = math.acos(-1.0)
		ndata = nshot
		nmp   = len(mp[np.where(mp[:,1]>=1.e-8),0][0])
		
		bic = ndata * (math.log(2.*pi) + 2.*math.log(rmstt) + 1.0) + float(nmp)*math.log(float(ndata))
		"""
		"""
		# print scatters
		sets = shot_dat['SET'].unique()
		shot_dat['ID'] = [ int(mt[1:]) for mt in shot_dat.MT]
		print(shot_dat.columns.values)
		
		nam = os.path.dirname(resf)+"/"+os.path.basename(obsfile.replace("obs.csv","set-"))
		for s in sets:
			df = shot_dat[shot_dat.SET == s].loc[:,['ID','TT','ResiTT','ST','ant_e0','ant_n0','ant_u0','head0','pitch0','roll0','dV0']]
			fs = 20
			
			axs = plotting.scatter_matrix(df.iloc[:, 1:], figsize=(fs, fs), c=list(-np.fabs(df.iloc[:, 0])), diagonal='kde', alpha=0.4)
			
			png = nam + s + ".png"
			plt.savefig(png)
			plt.close()
		"""
		"""
		# print sv spectrum
		for s in range(len(sets)):
			t = svtimeseries[s].t.values
			f = svtimeseries[s].sv0.values
			#print(np.unique(np.diff(t)))
			#print(np.unique(np.diff(t))[0] - np.unique(np.diff(t))[1])
			#print(f)
			
			spectra = Spectra(t/3600., f, 'Hour')
			plt = spectra.draw_with_time()
			plt.show()
			plt.close()
		
		
		t = np.array([])
		x = np.array([])
		for s in range(len(sets)):
			t = np.append(t,svtimeseries[s].t.values)
			x = np.append(x,svtimeseries[s].sv0.values)
		t = t/3600.
		f = x - np.poly1d(np.polyfit(t,x,1))(t)
		plt.plot(t,f)
		plt.plot(t,x)
		plt.show()
		plt.close()
		freq = np.linspace(0.05, 6, 1000)
		pgram = LS(t,f,freq, normalize=False,precenter=True)
		plt.plot(np.reciprocal(freq),pgram,lw=0.5)
		plt.show()
		plt.close()
		
		sys.exit()
		"""
	
	#return bic, [nmpsv0,nmpsv1,nmpsv2], ndata, nmp, mpnum, len(twin)
	return

if __name__ == '__main__':
	from optparse import OptionParser
	import os, sys, glob, configparser, datetime
	import numpy as np
	
	######################################################################
	# OptionParserを使ったオプションの設定
	usa = u"Usage: %prog [options] " # スクリプトの使用方法を表す文字列
	parser = OptionParser(usage=usa)
	parser.add_option("--site", action="store", type="string", default="SITE", dest="SITE", help=u'Set site name')
	parser.add_option("--posdiff", action="store_true", default=False, dest="posd", help=u"Make pos.diff file")
	parser.add_option("--resfiles", action="store", type="string", default="", dest="resfls", help=u"Set res-file names (if not set, all files in -d directory are used)")
	parser.add_option("--scale", action="store", type="string", default="0.30", dest="scale", help=u'Set_scale')
	(options, args) = parser.parse_args()
	#####################################################################
	
	resfiles = glob.glob(options.resfls)
	resfiles.sort()
	
	if len(resfiles) == 0:
		print("res.dat ファイル %s がありません" % options.resfls)
		sys.exit(1)
	if len(resfiles) == 1 and options.posd:
		print(" res.dat ファイル %s が 1 つしかありません" % resfiles)
		sys.exit(1)
	
	iresf = 0
	
	# make array geometry from resfiles
	allMT = []
	alldpos = []
	posdata = []
	ydate   = []
	dcpos = np.array([])
	
	for resf in resfiles:
		
		cfg = configparser.ConfigParser()
		cfg.read(resf, 'UTF-8')
		
		### Check site-name ###
		site  = cfg.get("Obs-parameter", "Site_name")
		#camp  = cfg.get("Obs-parameter", "Campaign")
		date1 = cfg.get("Obs-parameter", "Date(UTC)")
		date0 = cfg.get("Obs-parameter", "Date(jday)")
		year, day = date0.split("-")
		ydate.append(float(year) + float(day)/365.)
		if not site in options.SITE:
			print("Bad res-file (site-name does not match) : %s"  % resf)
			sys.exit(1)
		print(site, resf)
		
		### Check geocent ###
		geocent = [ float( cfg.get("Site-parameter", "Latitude0")  ),
					float( cfg.get("Site-parameter", "Longitude0") ),
					float( cfg.get("Site-parameter", "Height0")    ), 0 ]
		if not iresf == 0:
			if geocent != geocent0:
				print("Bad res-file (geometry-center does not match) : %s"  % resf)
				sys.exit(1)
		geocent0 = geocent
		
		### Read array-center ###
		MTs  = cfg.get("Site-parameter", "Stations").split(" ")
		MTs = [ str(mt) for mt in MTs]
		allMT += MTs
		
		# plot obsfile
		d0 = datetime.datetime.strptime(date1,"%Y-%m-%d")
		obsf = cfg.get("Data-file", "datacsv")
		mpf  = resf.replace('res.dat','m.p.dat')
		corf = resf.replace('res.dat','cor.dat')
		covf = resf.replace('res.dat','var.dat')
		
		#aic, nmpsv, ndata, nmp, mpnum, nset = plot_residuals(resf, obsf, mpf, d0, MTs)
		plot_residuals(resf, obsf, mpf, d0, MTs, options.scale)
		
		# plot corfile
		#plot_corMap(corf,covf,mpnum,nset)
		
		cntpos = cfg.get("Site-parameter","Center_ENU").split()
		cpos = np.array([])
		cpos = np.append(cpos, float(cntpos[0]))
		cpos = np.append(cpos, float(cntpos[1]))
		cpos = np.append(cpos, float(cntpos[2]))
		
		### Read station-positions ###
		dcntpos = cfg.get("Model-parameter", "dCentPos").split()
		dcpos = np.append(dcpos, float(dcntpos[0]))
		dcpos = np.append(dcpos, float(dcntpos[1]))
		dcpos = np.append(dcpos, float(dcntpos[2]))
		iresf += 1
		#print(resf, nmpsv, aic, ndata, nmp)
	
	nresf = iresf
	
	if options.posd:
		posdiff = "/".join(resfiles[0][:-1].split("/")[:-2]) + "/posdiff."+options.SITE+".dat"
		spd = open(posdiff, "w")
		spd.write("#Year          EW[m]       NS[m]       UD[m]\n")
		print(posdiff)
		#print dcpos
		for iresf in range(nresf):
			de = dcpos[iresf*3+0] - dcpos[0]
			dn = dcpos[iresf*3+1] - dcpos[1]
			du = dcpos[iresf*3+2] - dcpos[2]
			spd.write("%10.3f  %10.4f  %10.4f  %10.4f\n" % (ydate[iresf],de,dn,du))
			
		spd.close()
	
	exit()
	
