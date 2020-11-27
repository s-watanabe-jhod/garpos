"""
Created:
	07/01/2020 by S. Watanabe
"""
import os
import numpy as np

def write_cfg(of, obsp, datap, sitep, modelp, comment=""):
	"""
	Write site-paramter file.
	"""
	nn = float(len(modelp[0]))
	pe = np.array([ tp[1]/nn for tp in modelp[0] ]).sum() + modelp[1][0]
	pn = np.array([ tp[2]/nn for tp in modelp[0] ]).sum() + modelp[1][1]
	pu = np.array([ tp[3]/nn for tp in modelp[0] ]).sum() + modelp[1][2]
	
	Config  = "[Obs-parameter] "
	Config += "\n Site_name   = " + obsp[0]
	Config += "\n Campaign    = " + obsp[1]
	Config += "\n Date(UTC)   = " + obsp[2]
	Config += "\n Date(jday)  = " + obsp[3]
	Config += "\n Ref.Frame   = " + obsp[4]
	Config += "\n SoundSpeed  = " + obsp[5]
	
	Config += "\n\n[Data-file]"
	Config += "\n datacsv     = " + datap[0]
	Config += "\n N_shot      = %5d" % ( datap[1] )
	Config += "\n used_shot   = %5d" % ( datap[2] )
	
	Config += "\n\n[Site-parameter]"
	Config += "\n Latitude0   = %12.8f" % sitep[0]
	Config += "\n Longitude0  = %12.8f" % sitep[1]
	Config += "\n Height0     = %6.2f"  % sitep[2]
	Config += "\n Stations    = " + " ".join( [ tp[0] for tp in modelp[0] ] )
	Config += "\n# Array_cent :   'cntpos_E'  'cntpos_N'  'cntpos_U'"
	Config += "\n Center_ENU  =  %10.4f  %10.4f  %10.4f" % (pe,pn,pu)
	
	Config += "\n\n[Model-parameter]"
	Config += "\n# MT_Pos     :   'stapos_E'  'stapos_N'  'stapos_U'   "
	Config += "'sigma_E'   'sigma_N'   'sigma_U'   "
	Config += "'cov_NU'    'cov_UE'    'cov_EN'"
	for tp in modelp[0]:
		pl  = "  ".join(["%10.4f" % p for p in tp[1:][0:6]]) + "  "
		pl += "  ".join(["%10.3e" % p for p in tp[1:][6:]])
		Config += ("\n %s_dPos    =  " % tp[0]) + pl
	
	
	pl  = "  ".join(["%10.4f" % p for p in modelp[1][0:6]]) + "  "
	pl += "  ".join(["%10.3e" % p for p in modelp[1][6:]])
	Config += "\n dCentPos    =  " + pl
	
	Config += "\n# ANT_to_TD  :    'forward' 'rightward'  'downward'   "
	Config += "'sigma_F'   'sigma_R'   'sigma_D'   "
	Config += "'cov_RD'    'cov_DF'    'cov_FR'"
	pl  = "  ".join(["%10.4f" % p for p in modelp[2][0:6]]) + "  "
	pl += "  ".join(["%10.3e" % p for p in modelp[2][6:]])
	Config += "\n ATDoffset   =  " + pl
	
	print(of)
	
	Config += "\n\n"
	Config += comment
	
	o = open(of,"w")
	o.write(Config)
	o.close()
	
	return


def outresults(odir, suf, cfg, invtyp, imp0, slvidx0,
			   C, mp, shots, comment, MTs, mtidx, av):
	"""
	Output the results.
	"""
	### Observation parameter ###
	site = cfg.get("Obs-parameter", "Site_name")
	camp = cfg.get("Obs-parameter", "Campaign")
	date0 = cfg.get("Obs-parameter", "Date(UTC)")
	datej = cfg.get("Obs-parameter", "Date(jday)")
	refframe = cfg.get("Obs-parameter", "Ref.Frame")
	geocent = [ float( cfg.get("Site-parameter", "Latitude0")  ),
				float( cfg.get("Site-parameter", "Longitude0") ),
				float( cfg.get("Site-parameter", "Height0")    )]
	svpf = cfg.get("Obs-parameter", "SoundSpeed")
	
	# filenames to output
	filebase = site + "." + camp + suf
	savfile  = odir + filebase + "-obs.csv"
	resfile  = odir + filebase + "-res.dat"
	varfile  = odir + filebase + "-var.dat"
	mplfile  = odir + filebase + "-m.p.dat"
	
	obsfile  = cfg.get("Data-file", "datacsv")
	if obsfile == savfile:
		savfile = odir + filebase + "_mod-obs.csv"
		print("Warning : same file name for input/output obs.csv")
		print("  changed the output name to %s" % savfile)
	
	##################
	# Write CFG Data #
	##################
	ii = [1, 2, 0]
	jj = [2, 0, 1]
	imp = 0
	MTpos = []
	
	C0pos = np.zeros( (imp0[0],imp0[0]) )
	for i, ipos in enumerate(slvidx0):
		for j, jpos in enumerate(slvidx0):
			C0pos[ipos, jpos] = C[i,j]
	for mt in MTs:
		lmt = [mt]
		poscov = C0pos[imp:imp+3, imp:imp+3]
		for k in range(3):
			idx = mtidx[mt] + k
			lmt.append(mp[idx])
		lmt = lmt + [ poscov[i][i]**0.5 for i in range(3)]
		lmt = lmt + [ poscov[i][j] for i,j in zip(ii,jj) ]
		MTpos.append(lmt)
		imp += 3
	
	poscov = C0pos[imp:imp+3, imp:imp+3]
	dcpos = [mp[len(MTs)*3+0], mp[len(MTs)*3+1], mp[len(MTs)*3+2]]
	dcpos = dcpos + [ poscov[i][i]**0.5 for i in range(3)]
	dcpos = dcpos + [ poscov[i][j] for i,j in zip(ii,jj) ]
	imp += 3
	
	poscov = C0pos[imp:imp+3, imp:imp+3]
	pbias = [mp[len(MTs)*3+3], mp[len(MTs)*3+4], mp[len(MTs)*3+5]]
	pbias = pbias + [ poscov[i][i]**0.5 for i in range(3)]
	pbias = pbias + [ poscov[i][j] for i,j in zip(ii,jj) ]
	imp += 3
	
	resobsp   = [site, camp, date0, datej, refframe, svpf]
	resdatap  = [savfile, len(shots.index), shots[~shots['flag']].index.size]
	ressitep  = [geocent[0], geocent[1], geocent[2]]
	resmodelp = [MTpos, dcpos, pbias]
	
	write_cfg(resfile, resobsp, resdatap, ressitep, resmodelp, comment)
	
	###################
	# Write Shot Data #
	###################
	if invtyp != 0:
		shots['dV0'] = av[0][0]/2. + av[1][0]/2.
		shots['gradV1e'] = av[0][1]/2. + av[1][1]/2.
		shots['gradV1n'] = av[0][2]/2. + av[1][2]/2.
		shots['gradV2e'] = av[0][3]/2. + av[1][3]/2.
		shots['gradV2n'] = av[0][4]/2. + av[1][4]/2.
	elif not "dV0" in shots.columns:
		shots['dV0'] = 0.
		shots['gradV1e'] = 0.
		shots['gradV1n'] = 0.
		shots['gradV2e'] = 0.
		shots['gradV2n'] = 0.
	
	shots["LogResidual"] = shots["ResiTT"]
	shots["ResiTT"] = shots["ResiTTreal"]
	ashot = shots.loc[:,[
		'SET','LN','MT','TT','ResiTT', 'TakeOff', 'gamma', 'flag',
		'ST','ant_e0','ant_n0','ant_u0','head0','pitch0','roll0',
		'RT','ant_e1','ant_n1','ant_u1','head1','pitch1','roll1',
		'dV0', 'gradV1e', 'gradV1n', 'gradV2e', 'gradV2n',
		'dV', 'LogResidual', 
		]]
	
	ashot['TT'] = ashot['TT'].round(7)
	ashot['ST'] = ashot['ST'].round(7)
	ashot['RT'] = ashot['RT'].round(7)
	ashot['ant_e0'] = ashot['ant_e0'].round(5)
	ashot['ant_n0'] = ashot['ant_n0'].round(5)
	ashot['ant_u0'] = ashot['ant_u0'].round(5)
	ashot['ant_e1'] = ashot['ant_e1'].round(5)
	ashot['ant_n1'] = ashot['ant_n1'].round(5)
	ashot['ant_u1'] = ashot['ant_u1'].round(5)
	ashot['head0']  = ashot['head0'].round(2)
	ashot['roll0']  = ashot['roll0'].round(2)
	ashot['pitch0'] = ashot['pitch0'].round(2)
	ashot['head1']  = ashot['head1'].round(2)
	ashot['roll1']  = ashot['roll1'].round(2)
	ashot['pitch1'] = ashot['pitch1'].round(2)
	
	ashot.to_csv( savfile )
	hd = "# cfgfile = %s\n" % resfile
	outfile = open(savfile, "w")
	outfile.write(hd)
	outfile.close()
	ashot.to_csv(savfile, mode='a')
	
	######################################################
	# Write Covariance Matrix and Model Parameter Vector #
	######################################################
	np.savetxt(varfile, C, delimiter=',', fmt ='%.6e')
	
	sigmap = np.diag(C0pos)**0.5
	sigmag = np.diag(C)[len(slvidx0):]**0.5
	sigma = np.append(sigmap, sigmag)
	
	mptyp = np.zeros(len(mp))
	mpflg = np.zeros(len(mp))
	for k in range(5):
		mptyp[imp0[k]:imp0[k+1]] = k+1
		mpflg[imp0[k]:imp0[k+1]] = imp0[k+1]-imp0[k]
	
	mplist = np.array([mp, sigma, mptyp, mpflg])
	np.savetxt(mplfile, mplist.T, delimiter=',', fmt ='%16.6f')
	
	return resfile, dcpos
