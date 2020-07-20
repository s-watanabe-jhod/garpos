#!/bin/bash -eu
# -e エラーがあったらシェルスクリプトをそこで打ち止めにしてくれる
# -u 未定義の変数を使おうとしたときに打ち止めにしてくれる

#sitelist="HYG2"
#sitelist="HYG2 ASZ2 ASZ1 TOS2 TOS1 TOS2 MRT1 MRT2 SIOW KUM3 KUM2 KUM1 TOK3 TOK2 TOK1 SAGA BOSN CHOS FUKU MYGW MYGI KAMS KAMN"
sitelist="TU17 MYGI KAMS KAMN TU08 TU10 TU12 TU14"
#sitelist="MRT3"

finiprep="Settings-prep.ini"
finifix="Settings-fix.ini"
fininoise="Settings-noise.ini"

for site in ${sitelist}
do
#	for f2 in `ls -r ./demo_prep/${site}/${site}.1505*.*yo*obs.csv`
	for f2 in `ls -r ./demo_prep/${site}/${site}.[0-1][0-9]*.*yo*-obs.csv`
	do
		epoch=${f2:22:4}
                cp ${f2} testfile
		bn=`basename $f2`
		cfgfix_ini="cfgfix/${site}/${bn/-obs.csv}-fix.ini"
                cp ${cfgfix_ini} tes2file
		./specgram9.py > noneed.dat
		cp figure.png specgram/${site}_${epoch}.png
		echo $site $epoch
	done
#	for f2 in `ls -r ./demo_res/${site}/${site}.1505*.*yo*01.0-obs.csv`
	for f2 in `ls -r ./demo_res/${site}/${site}.[0-1][0-9]*.*yo*01.0-obs.csv`
	do
		epoch=${f2:21:4}
                cp ${f2} testfile
		bn=`basename $f2`
		cfgfix_ini="cfgfix/${site}/${bn/_L-01.0-obs.csv}-fix.ini"
                cp ${cfgfix_ini} tes2file
		./specgram9.py > noneed.dat
		cp figure.png specgram_fix/${site}_${epoch}.png
		echo $site $epoch
	done
done
rm noneed.dat
