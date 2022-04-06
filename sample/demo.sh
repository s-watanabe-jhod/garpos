#!/bin/bash -eu
bindir="../bin"
site="SAGA"

singledir="demo_prep"
cfgfixdir="cfgfix"
fixresdir="demo_res"

fini0="Settings-prep.ini"
fini1="Settings-fix.ini"

mc="1" # max core

### change for your correct python3 executable 
### or set "" to use [#!/usr/bin/env python]
#python_exe="python"
#python_exe="python3"
python_exe=""

slvsingle="${python_exe} ${bindir}/solveSingleEpoch.py"

# if false, skip #
isingle=true
ifix=true

# if false, view mode #
irun=true

# make result plots #
iplot=true


################
# solve single #
################
if "$isingle"; then
	for f in `ls ./initcfg/${site}/${site}.[0-2][0-9]*.*yo*initcfg.ini`
	do
		declare -a f1=()
		f1=$(echo $f | tr '/' ' ')
		for a in ${f1[@]}; do f2=$a; done
		
		bn=`basename $f`
		obs="${singledir}/${site}/${bn/initcfg.ini}obs.csv"
		res="${singledir}/${site}/${bn/initcfg.ini}res.dat"
		
		# for single Parameter set
		cmd="${slvsingle} -f ${f}  -i ${fini0} -d ./${singledir}/${site}" 
		echo ${cmd}
		
		if ls ${obs} > /dev/null 2>&1; then
		if ls ${res} > /dev/null 2>&1; then
			echo "*** SKIP: Single-Result (${obs} and ${res}) exists... ***"
			continue
		fi
		fi
		
		if "${irun}"; then ${cmd}; fi
		if [ $? = 1 ]; then echo "error!"; exit ; fi
		
		plotcmd="${python_exe} ${bindir}/plot_EpochResults_v1.0.py --resfiles \"./${singledir}/${site}/${bn/-initcfg.ini}*-res.dat\""
		
		if "${iplot}"; then echo ${plotcmd}; ${plotcmd}; fi
	
	done
	
	### make fixcfg ###
	cmd="${python_exe} ${bindir}/makeFixCfg.py -d ./${cfgfixdir}/ --res_singles \"./${singledir}/${site}/*-res.dat\""
	echo ${cmd}
	
	if "${irun}"; then ${cmd}; fi
	if [ $? = 1 ]; then echo "error!"; exit ; fi
fi

###########
# run fix #
###########
if "$ifix"; then
	for f in `ls ./${cfgfixdir}/${site}/${site}.*.*yo*fix.ini`
	do
		declare -a f1=()
		f1=$(echo $f | tr '/' ' ')
		for a in ${f1[@]}; do f2=$a; done
		
		bn=`basename $f`
		obs=${fixresdir}/${site}/${bn/-fix.ini}*obs.csv
		
		if ls ${obs} > /dev/null 2>&1; then
			echo "*** SKIP: Single-Result (${obs}) exists... ***"
			continue
		fi
		
		# for fix solve with search Lambda
		cmd="${slvsingle} -f ${f}  -i ${fini1} -d ./${fixresdir}/${site} --maxcore ${mc}"
		echo ${cmd}
		
		if "${irun}"; then ${cmd}; fi
		if [ $? = 1 ]; then echo "error!"; exit ; fi
		
		plotcmd="${python_exe} ${bindir}/plot_EpochResults_v1.0.py --resfiles \"./${fixresdir}/${site}/${bn/-fix.ini}*-res.dat\""
		
		if "${iplot}"; then echo ${plotcmd}; ${plotcmd}; fi
	done
	
	cmd="${python_exe} ${bindir}/makePosDiff.py --site ${site} -d ./${fixresdir} --resfiles \"./${fixresdir}/${site}/*-res.dat\""
	echo ${cmd}
	if "${irun}"; then ${cmd}; fi
fi

