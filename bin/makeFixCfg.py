#!/usr/bin/env python
import os
import sys
import glob
import configparser
from optparse import OptionParser
import numpy as np
import pandas as pd


if __name__ == '__main__':
	######################################################################
	usa = u"Usage: %prog [options] "
	opt = OptionParser(usage=usa)
	opt.add_option( "-d", action="store", type="string",
					default="./cfg_fix/", dest="directory",
					help=u"Set save directory"
					)
	opt.add_option( "--res_singles", action="store", type="string",
					default="", dest="resfiles",
					help=u"Single-epoch result files"
					)
	(options, args) = opt.parse_args()
	#####################################################################
	
	resfiles = options.resfiles.strip('"').strip("'")
	resfiles = glob.glob(resfiles)
	resfiles.sort()
	nresf = len(resfiles)
	
	rmcri = 4.0
	
	if nresf == 0:
		print("NOT FOUND (site-parameter files) :: %s" % options.resfiles)
		sys.exit(1)
	elif nresf == 1:
		print("More than 2 files needed :: %s" % options.resfiles)
		sys.exit(1)
	
	# make array geometry from resfiles
	allMT = []
	alldpos = []
	posdata = []
	epochMT = []
	
	cfg = configparser.ConfigParser()
	cfg.read(resfiles[0], 'UTF-8')
	site0  = cfg.get("Obs-parameter", "Site_name")
	
	for iresf, resf in enumerate(resfiles):
		
		cfg = configparser.ConfigParser()
		cfg.read(resf, 'UTF-8')
		
		### Verify the input files ###
		site  = cfg.get("Obs-parameter", "Site_name")
		date0 = cfg.get("Obs-parameter", "Date(jday)")
		if site != site0:
			print("Bad res-file (site-name mismatch) : %s"  % resf)
			sys.exit(1)
		
		geocent = [ float( cfg.get("Site-parameter", "Latitude0")  ),
					float( cfg.get("Site-parameter", "Longitude0") ),
					float( cfg.get("Site-parameter", "Height0")    )]
		if not iresf == 0:
			if geocent != geocent0:
				print("Bad res-file (ref. pos. mismatch) : %s"  % resf)
				sys.exit(1)
		geocent0 = geocent
		
		dpos = cfg.get("Model-parameter", "dCentPos").split()
		dpos = list(map(float, dpos))
		if dpos[0]**2. + dpos[1]**2. + dpos[2]**2. > 0.001:
			print("Bad res-file (dCentPos /= 0.) : %s" % resf)
			sys.exit(1)
		
		### make directory ###
		if iresf == 0:
			cfgdir = options.directory+"/"+site+"/"
			if not os.path.exists(cfgdir):
				os.makedirs(cfgdir[:-1])
		
		### Read array-center ###
		MTs  = cfg.get("Site-parameter", "Stations").split(" ")
		MTs = [ str(mt) for mt in MTs]
		allMT += MTs
		epochMT.append(MTs)
		cpos = cfg.get("Site-parameter","Center_ENU").split()
		cpos = np.array(list(map(float, cpos)))
		
		### Read station-positions ###
		mtpos = {}
		for mt in MTs:
			pos = cfg.get("Model-parameter", mt + "_dPos").split()
			pos = list(map(float, pos))
			mp = np.array(pos[0:3])
			er = np.array(pos[3:6])
			
			# write into dictionary
			mtpos[mt] = mp
			posdata.append([mt, iresf, mp, er])
		alldpos.append(mtpos)
		
	
	allMT = list(set(allMT))
	allMT.sort()
	nmt   = len(allMT)
	
	#################################
	# Calc. averaged array geometry #
	#################################
	pdata = pd.DataFrame(posdata, columns=["MT","camp","pos","err"])
	pdata["flag"] = False
	comment = ""
	flaglist = []
	for i in range(30):
		pdata = pdata[ ~pdata.flag ].reset_index(drop=True)
		
		camp = pdata.camp
		idmt = pdata.MT
		
		allcamp = list(set(camp))
		allcamp.sort()
		ncamp = len(allcamp)
		
		ndata = len(pdata)
		npara = ncamp + nmt
		
		H = np.zeros((ndata+1, npara))
		data = np.zeros((ndata+1,3))
		
		for idx, mt in enumerate(idmt):
			icamp = allcamp.index(camp[idx])
			imt = allMT.index(mt)
			
			H[idx][imt] = 1.
			H[idx][nmt+icamp] = 1.
			data[idx][:] = pdata.pos.values[idx]
		
		H[ndata][nmt:] = 1.
		
		HtH    = np.matmul(H.T, H)
		HtHi   = np.linalg.inv(HtH)
		HtHiHt = np.matmul(HtHi, H.T)
		para = np.matmul(HtHiHt, data)
		
		calc = np.matmul(H, para)[:-1]
		obsd = data[:-1]
		depth = np.mean(obsd[:,2])
		
		base = obsd/abs(depth)
		
		oc = obsd-calc
		
		rms = lambda d: np.sqrt((d ** 2.).sum() / d.size)
		erms = rms(oc[:,0])
		nrms = rms(oc[:,1])
		urms = rms(oc[:,2])
		
		comment += "# RMS(E,N,U) = %8.4f %8.4f %8.4f\n" % (erms,nrms,urms)
		comment += "# Used epochs: "
		comment += " ".join(map(str,allcamp))
		comment += "\n"
		fl0 = len(flaglist)
		
		for i, denu in enumerate(oc):
			hd = "%03d %s " % (camp[i],idmt[i])
			rf = resfiles[camp[i]]
			fl = False
			if abs(denu[0]) > rmcri*erms:
				fl = True
				v = denu[0]
				comment += "#  %s E %6.3f %4.1f-sigma %s\n" % (hd,v,v/erms,rf)
			if abs(denu[1]) > rmcri*nrms:
				fl = True
				v = denu[1]
				comment += "#  %s N %6.3f %4.1f-sigma %s\n" % (hd,v,v/nrms,rf)
			if abs(denu[2]) > rmcri*urms:
				fl = True
				v = denu[2]
				comment += "#  %s U %6.3f %4.1f-sigma %s\n" % (hd,v,v/urms,rf)
			if fl:
				flaglist.append(camp[i])
		flaglist = list(set(flaglist))
		
		pdata["flag"] = pdata.camp.isin(flaglist)
		
		comment += "# \n"
		
		if len(flaglist)-fl0 == 0:
			break
	
	comment += "# Excluded epochs for array geometry ::\n"
	for i in flaglist:
		comment += "#  " + resfiles[i] + "\n"
		comment += "#   " + " ".join(epochMT[i]) + "\n"
	
	print(comment)
	
	# modify res file to ini file
	pe = para[0:nmt,0].mean()
	pn = para[0:nmt,1].mean()
	pu = para[0:nmt,2].mean()
	CENU = " Center_ENU  =  %10.4f  %10.4f  %10.4f" % (pe,pn,pu)
	
	imt = 0
	dPos = []
	for mt in allMT:
		pl = ""
		for p in para[imt]:
			pl += "  %10.4f" % p
		line  = (" %s_dPos    =" % mt) + pl
		line += "  %10.4f  %10.4f  %10.4f" % (0.,0.,0.)
		dPos.append(line)
		imt += 1
	
	logfile = options.directory+"/" + "mkcfg." + site0 + ".log"
	olog = open(logfile,"w")
	olog.write(comment)
	olog.close()
	
	for iresf, resf in enumerate(resfiles):
		dCPos  = " dCentPos    =  %10.4f  %10.4f  %10.4f" % (0.,0.,0.)
		dCPos += "  %10.4f  %10.4f  %10.4f" % (3.,3.,3.)
		
		basename = os.path.basename(resf)
		cfgf = os.path.join(cfgdir, basename.split("-")[0] + "-fix.ini")
		
		# modify ini file
		f = open(resf,"r")
		o = open(cfgf,"w")
		ol = []
		for l in f:
			pl = l
			if l.find("Center_ENU") != -1:
				pl = CENU+"\n"
			elif l.find("dCentPos") != -1:
				pl = dCPos+"\n"
			for imt in range(len(allMT)):
				s = "%s_dPos" % allMT[imt]
				if l.find(s) != -1:
					pl = dPos[imt]+"\n"
					break
			ol.append(pl)
		
		for l in ol:
			o.write(l)
		f.close()
		o.close()
	
	exit()
	
