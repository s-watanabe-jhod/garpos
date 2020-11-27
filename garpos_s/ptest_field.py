#!/usr/bin/python_ana3
# -*- coding: utf-8 -*-
############################################
import numpy as np
import pandas as pd
import shutil
import datetime
import os,glob
import matplotlib.pyplot as plt
#from garpos.itrf_trans_posdiff import itrf_trans

figdir="./demo_res/FUKU/fig"
obsfile=glob.glob("./demo_res/FUKU/FUKU.*1806*-obs.csv")
#obsfile = "./demo_res/FUKU/FUKU.1806.kaiyo_k4-obs.csv"
for nl in range(len(obsfile)):
	obsimg = figdir+"/"+os.path.basename(obsfile[nl].replace("obs.csv","fld.png"))
	shot_dat = pd.read_csv(obsfile[nl], comment='#', index_col=0)
	fig = plt.figure(figsize=(10,40))
	ax0 = fig.add_subplot(4, 1, 1) 
	ax1 = fig.add_subplot(4, 1, 2) 
	ax2 = fig.add_subplot(4, 1, 3) 
	ax3 = fig.add_subplot(4, 1, 4) 
	plt.rcParams["font.size"] = 12
	depth=1250.
	scale =depth*1.05
	scales=scale/2
	#plt.set_xlim( -scalel, scalel)
	#plt.set_ylim( -scalel, scalel)
	#plt.set_aspect('equal')
	setcmap = plt.get_cmap("Set3")
	inode=5
	e1=[]
	n1=[]
	s1=[]
	for j in range(inode):
		for i in range(inode):
			e1.append((-1.*(inode-1)*0.5+float(i))*scales)
			n1.append((-1.*(inode-1)*0.5+float(j))*scales)
			s1.append("gsp0_"+str(i+inode*j))
	e2=[]
	n2=[]
	s2=[]
	for j in range(inode-2):
		for i in range(inode-2):
			e2.append((-1.*(inode-1)*0.5+2*float(i))*scales)
			n2.append((-1.*(inode-1)*0.5+2*float(j))*scales)
			s2.append("gsp1_"+str(i+(inode-2)*j))
	#for t in range(len(shot_dat)):
	ax0.plot(shot_dat.ant_e0, shot_dat.ant_n0, marker='o', markersize=2, linestyle='None')
	ax1.plot(shot_dat.ant_e0, shot_dat.ant_n0, marker='o', markersize=2, linestyle='None')
	ax2.plot(shot_dat.ant_e0, shot_dat.ant_n0, marker='o', markersize=2, linestyle='None')
	ax3.plot(shot_dat.ant_e0, shot_dat.ant_n0, marker='o', markersize=2, linestyle='None')
	for j in range(inode):
		for i in range(inode):
			s=int(len(shot_dat)/5)
			ax0.text(e1[i+inode*j],n1[i+inode*j], '{:.3f}'.format(shot_dat[s1[i+inode*j]][s]), fontsize=15)
			ax0.plot(e1[i+inode*j],n1[i+inode*j], color=setcmap(shot_dat[s1[i+inode*j]][s]), marker='s', markersize=20, linestyle='None')
			s=int(len(shot_dat)*2/5)
			ax1.text(e1[i+inode*j],n1[i+inode*j], '{:.3f}'.format(shot_dat[s1[i+inode*j]][s]), fontsize=15)
			ax1.plot(e1[i+inode*j],n1[i+inode*j], color=setcmap(shot_dat[s1[i+inode*j]][s]), marker='s', markersize=20, linestyle='None')
			s=int(len(shot_dat)*3/5)
			ax2.text(e1[i+inode*j],n1[i+inode*j], '{:.3f}'.format(shot_dat[s1[i+inode*j]][s]), fontsize=15)
			ax2.plot(e1[i+inode*j],n1[i+inode*j], color=setcmap(shot_dat[s1[i+inode*j]][s]), marker='s', markersize=20, linestyle='None')
			s=int(len(shot_dat)*4/5)
			ax3.text(e1[i+inode*j],n1[i+inode*j], '{:.3f}'.format(shot_dat[s1[i+inode*j]][s]), fontsize=15)
			ax3.plot(e1[i+inode*j],n1[i+inode*j], color=setcmap(shot_dat[s1[i+inode*j]][s]), marker='s', markersize=20, linestyle='None')
	for j in range(inode-2):
		for i in range(inode-2):
			s=int(len(shot_dat)/5)
			ax0.text(e2[i+(inode-2)*j],n2[i+(inode-2)*j]-scales*0.2, '{:.3f}'.format(shot_dat[s2[i+(inode-2)*j]][s]), fontsize=15)
			ax0.plot(e2[i+(inode-2)*j],n2[i+(inode-2)*j]-scales*0.2, color=setcmap(shot_dat[s2[i+(inode-2)*j]][s]), marker='s', markersize=20, linestyle='None')
			s=int(len(shot_dat)*2/5)
			ax1.text(e2[i+(inode-2)*j],n2[i+(inode-2)*j]-scales*0.2, '{:.3f}'.format(shot_dat[s2[i+(inode-2)*j]][s]), fontsize=15)
			ax1.plot(e2[i+(inode-2)*j],n2[i+(inode-2)*j]-scales*0.2, color=setcmap(shot_dat[s2[i+(inode-2)*j]][s]), marker='s', markersize=20, linestyle='None')
			s=int(len(shot_dat)*3/5)
			ax2.text(e2[i+(inode-2)*j],n2[i+(inode-2)*j]-scales*0.2, '{:.3f}'.format(shot_dat[s2[i+(inode-2)*j]][s]), fontsize=15)
			ax2.plot(e2[i+(inode-2)*j],n2[i+(inode-2)*j]-scales*0.2, color=setcmap(shot_dat[s2[i+(inode-2)*j]][s]), marker='s', markersize=20, linestyle='None')
			s=int(len(shot_dat)*4/5)
			ax3.text(e2[i+(inode-2)*j],n2[i+(inode-2)*j]-scales*0.2, '{:.3f}'.format(shot_dat[s2[i+(inode-2)*j]][s]), fontsize=15)
			ax3.plot(e2[i+(inode-2)*j],n2[i+(inode-2)*j]-scales*0.2, color=setcmap(shot_dat[s2[i+(inode-2)*j]][s]), marker='s', markersize=20, linestyle='None')
	plt.savefig(obsimg)
	plt.close()
