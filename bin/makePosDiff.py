#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################

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
	parser = OptionParser(usage=usa)
	parser.add_option( "--site", action="store", type="string", default = "SITE", dest="SITE", help=u'Set site name' )
	parser.add_option( "-d", action="store", type="string", default="./posdiff_single/", dest="directory", help=u"Set directory for output" )
	parser.add_option( "--resfiles", action="store", type="string", default="", dest="resfls", help=u"Result files")
	parser.add_option( "--suffix", action="store", type="string", default="", dest="suf", help=u"Set suffix for posdiff files (res.SITE[suf].dat)" )
	(options, args) = parser.parse_args()
	#####################################################################

	if options.resfls=="":
		print("Use --resfiles option for 'res.dat' in single-epoch analysis")
		sys.exit(1)

	resfs = options.resfls.strip('"').strip("'")
	resfiles = glob.glob(resfs)
	resfiles.sort()

	if len(resfiles) == 0:
		print("Not found (res.dat) : %s " % options.res_files)
		sys.exit(1)
	elif len(resfiles) == 1:
		print("Only 1 res.dat file is found : %s " % options.res_files)
		sys.exit(1)

	# make array geometry from resfiles
	allMT = []
	alldpos = []
	posdata = []
	ydate   = []

	### make directory ###
	pddir = options.directory+"/"
	if not os.path.exists(pddir):
		os.makedirs(pddir[:-1])
	posdiff = options.directory + "/res."+options.SITE+options.suf+".dat"

	posd = open(posdiff, "w")
	posd.write("#SITE: %s\n" % options.SITE)
	posd.write("#Year          EW[m]      NS[m]      UD[m]   sgmEW[m]   sgmNS[m]   sgmUD[m]\n")
	print(posdiff)

	#check observation date
	dd = []
	for resf in resfiles:
		cfg = configparser.ConfigParser()
		cfg.read(resf, 'UTF-8')
		date0 = cfg.get("Obs-parameter", "Date(jday)")
		year, day = date0.split("-")
		date = (float(year)-2000.) + float(day)/365.
		dd.append(date)
	rf = pd.DataFrame(list(zip(*[dd, resfiles])), columns=['d', 'file'])
	rf = rf.sort_values('d', ascending=True)

	iresf = 0
	for resf in rf["file"]:

		cfg = configparser.ConfigParser()
		cfg.read(resf, 'UTF-8')

		### Check site-name ###
		site  = cfg.get("Obs-parameter", "Site_name")
		date0 = cfg.get("Obs-parameter", "Date(jday)")
		year, day = date0.split("-")
		yy = float(year) + float(day)/365.
		ydate.append(yy)
		if site != options.SITE:
			print("Bad res-file (site-name does not match) : %s"  % resf)
			sys.exit(1)

		### Check geocent ###
		geocent = [ float( cfg.get("Site-parameter", "Latitude0")  ),
					float( cfg.get("Site-parameter", "Longitude0") ),
					float( cfg.get("Site-parameter", "Height0")    )]
		if not iresf == 0:
			if geocent != geocent0:
				print("Bad res-file (geometry-center does not match) : %s"  % resf)
				sys.exit(1)
		geocent0 = geocent

		### Check dCentPos ###
		dpos = cfg.get("Model-parameter", "dCentPos").split()
		dpos = np.array([yy, float(dpos[0]), float(dpos[1]), float(dpos[2]),float(dpos[3]), float(dpos[4]), float(dpos[5])])
		ldp = " ".join(['{:10.4f}'.format(a) for a in dpos])
		posd.write(ldp+"\n")
		print(ldp)

		iresf += 1

	posd.close()

	exit()
