#!/usr/bin/python3
# -*- coding: utf-8 -*-
############################################

from scipy import fftpack
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
import statistics

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

def plot_residuals(resf, obsfile, mpfile, d0, MTs):
	
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
			dVimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","dvcor.png"))
			dVcorcsv = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","dvcor.csv"))
			dVcsv = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","dv.csv"))

		else:
			#acfimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","acf%1d.png"%(isetg)))
			obsimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","t.s%1d.png"%(isetg)))
			dVimg = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","dvcor%1d.png"%(isetg)))
			dVcorcsv = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","dvcor%1d.csv"%(isetg)))
			dVcsv = figdir+"/"+os.path.basename(obsfile.replace("obs.csv","dv%1d.csv"%(isetg)))
		
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
		
		for i in range(len(MTs)):
			mt = MTs[i]
			xx = shot_tmp.loc[(shot_tmp['MT']==mt),'dt']
			yy = shot_tmp.loc[(shot_tmp['MT']==mt),'ResiTT']
			
			xr = rejected.loc[(rejected['MT']==mt),'dt']
			yr = [len(MTs)-i for x in xr]
		
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
		
		## dSV ##
		y0 = shot_tmp['dV0'].min()
		y1 = shot_tmp['dV0'].max()
		y0 = min(float(math.ceil(y0*10.))/10., float(math.floor(y0*10.))/10.)
		y1 = max(float(math.ceil(y1*10.))/10., float(math.floor(y1*10.))/10.)

		for i in range(len(MTs)):
			mt = MTs[i]
			xx = shot_tmp.loc[(shot_tmp['MT']==mt),'dt']
			yy = shot_tmp.loc[(shot_tmp['MT']==mt),'dV']
		
		## grad SV ##
		offset1 =  1.0
		offset2 = -1.0
		
		gradVs = [ ((e+e2)**2.+(n+n2)**2.)**0.5 for e, n, e2, n2 in zip(shot_tmp.gradV1e,shot_tmp.gradV1n,shot_tmp.gradV2e,shot_tmp.gradV2n)]
		gradVd = [ (e**2.+n**2.)**0.5 for e, n in zip(shot_tmp.gradV2e,shot_tmp.gradV2n)]
		
		scaleVs = float(math.ceil(max(gradVs)*10.))/10.
		scaleVd = float(math.ceil(max(gradVd)*10.))/10.
		scale = max(scaleVs, scaleVd)
####
		## g2 + g1 * i
#		gradVe = [ (s**2.+d**2.)**0.5 for s, d in zip(shot_tmp.gradV1e,shot_tmp.gradV2e)]
#		gradVn = [ (s**2.+d**2.)**0.5 for s, d in zip(shot_tmp.gradV1n,shot_tmp.gradV2n)]
#		radVe  = [ math.atan2(d,s)    for s, d in zip(shot_tmp.gradV1e,shot_tmp.gradV2e)]
#		radVn  = [ math.atan2(d,s)    for s, d in zip(shot_tmp.gradV1n,shot_tmp.gradV2n)]
		gradVe = [ ((s+d)**2.+(d)**2.)**0.5 for s, d in zip(shot_tmp.gradV1e,shot_tmp.gradV2e)]
		gradVn = [ ((s+d)**2.+(d)**2.)**0.5 for s, d in zip(shot_tmp.gradV1n,shot_tmp.gradV2n)]
		radVe  = [ math.cos(math.atan2(s+d,d)) for s, d in zip(shot_tmp.gradV1e,shot_tmp.gradV2e)]
		radVn  = [ math.cos(math.atan2(s+d,d)) for s, d in zip(shot_tmp.gradV1n,shot_tmp.gradV2n)]
#		print(statistics.mean(gradVe),statistics.pstdev(gradVe))
#		print(statistics.mean(gradVn),statistics.pstdev(gradVn))
#		print(statistics.mean(radVe),statistics.pstdev(radVe))
#		print(statistics.mean(radVn),statistics.pstdev(radVn))
		dV_zero = shot_tmp['dV0']
		dV_diff = np.diff(dV_zero)
		dV_diff = np.append(dV_diff, dV_diff[-1])
		shot_tmp = shot_tmp.drop("SET", axis=1)
		shot_tmp = shot_tmp.drop("LN", axis=1)
		shot_tmp = shot_tmp.drop("MT", axis=1)
		shot_tmp = shot_tmp.drop("TT", axis=1)
		shot_tmp = shot_tmp.drop("ResiTT", axis=1)
		shot_tmp = shot_tmp.drop("TakeOff", axis=1)
		shot_tmp = shot_tmp.drop("gamma", axis=1)
		shot_tmp = shot_tmp.drop("flag", axis=1)
		shot_tmp = shot_tmp.drop("ST", axis=1)
		shot_tmp = shot_tmp.drop("ant_e0", axis=1)
		shot_tmp = shot_tmp.drop("ant_n0", axis=1)
		shot_tmp = shot_tmp.drop("ant_u0", axis=1)
		shot_tmp = shot_tmp.drop("head0", axis=1)
		shot_tmp = shot_tmp.drop("pitch0", axis=1)
		shot_tmp = shot_tmp.drop("roll0", axis=1)
		shot_tmp = shot_tmp.drop("RT", axis=1)
		shot_tmp = shot_tmp.drop("ant_e1", axis=1)
		shot_tmp = shot_tmp.drop("ant_n1", axis=1)
		shot_tmp = shot_tmp.drop("ant_u1", axis=1)
		shot_tmp = shot_tmp.drop("head1", axis=1)
		shot_tmp = shot_tmp.drop("pitch1", axis=1)
		shot_tmp = shot_tmp.drop("roll1", axis=1)
		shot_tmp = shot_tmp.drop("gradV1e", axis=1)
		shot_tmp = shot_tmp.drop("gradV1n", axis=1)
		shot_tmp = shot_tmp.drop("gradV2e", axis=1)
		shot_tmp = shot_tmp.drop("gradV2n", axis=1)
		shot_tmp = shot_tmp.drop("dV", axis=1)
		shot_tmp = shot_tmp.drop("LogResidual", axis=1)
		shot_tmp = shot_tmp.drop("dt", axis=1)
		shot_tmp = shot_tmp.drop("zeros", axis=1)
		shot_tmp = shot_tmp.drop("a", axis=1)
		shot_tmp = shot_tmp.assign(absolute_E=gradVe, absolute_N=gradVn, angle_E=radVe, angle_N=radVn)
#		shot_tmp = shot_tmp.assign(A_Vd_iVs_E=gradVe, A_Vd_iVs_N=gradVn, Vd_iVs_E=radVe, Vd_iVs_N=radVn, vel_dV0=dV_diff)
#		shot_tmp = shot_tmp[['dV0','vel_dV0','gradV1e','gradV1n','gradV2e','gradV2n','A_Vd_iVs_E','A_Vd_iVs_N','Vd_iVs_E','Vd_iVs_N']]
		res=shot_tmp.corr()
#		print(res)
#		print(shot_tmp)
		mp0 = [0]
		mp1 = [0]
		sta = []
		s0  = []
		s1  = []
		fig = plt.figure(figsize=(14,12))
		colormap = generate_cmap(['navy', 'royalblue', 'cornflowerblue', '#EEEEEE', '#EEEEEE', '#EEEEEE', 'cornflowerblue', 'royalblue', 'navy'])
		plt.title('Correlation of $g_2+g_1$ with $dV$', y=1.05, size=15)
		ax = sns.heatmap(res, cmap=colormap, annot=True,fmt='.2f', vmin = -1, vmax = 1)
		res.to_csv(dVcorcsv)
	
		for m in mp0[1:-1]:
			ax.plot(oline, [m,m], 'g-', lw = 1)
			ax.plot([m,m], oline, 'g-', lw = 1)
		for s in sta:
			ax.plot(oline, [s,s], 'y-', lw = 0.5)
			ax.plot([s,s], oline, 'y-', lw = 0.5)
		for s in s0:
			ax.plot(oline, [s,s], 'y-', lw = 0.5)
			ax.plot([s,s], oline, 'y-', lw = 0.5)
		plt.savefig(dVimg)
		plt.close()
	
		
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
		plot_residuals(resf, obsf, mpf, d0, MTs)
	
	exit()
	
