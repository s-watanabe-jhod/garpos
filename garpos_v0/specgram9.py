#!/home/sgo/anaconda3/bin/python3
import os
import sys
import glob
import configparser
from optparse import OptionParser
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
ip1 = ["最近傍点補間", lambda x, y: interpolate.interp1d(x, y, kind="nearest")]
ip2 = ["線形補間", interpolate.interp1d]
ip3 = ["ラグランジュ補間", interpolate.lagrange]
ip4 = ["重心補間", interpolate.BarycentricInterpolator]
ip5 = ["Krogh補間", interpolate.KroghInterpolator]
ip6 = ["2次スプライン補間", lambda x, y: interpolate.interp1d(x, y, kind="quadratic")]
ip7 = ["3次スプライン補間", lambda x, y: interpolate.interp1d(x, y, kind="cubic")]
ip8 = ["秋間補間", interpolate.Akima1DInterpolator]
ip9 = ["区分的 3 次エルミート補間", interpolate.PchipInterpolator]
####
obsfile = "testfile"
cfgfile = "tes2file"
cfg = configparser.ConfigParser()
cfg.read(cfgfile, 'UTF-8')
fig,axes = plt.subplots(nrows=4,ncols=4,figsize=(10,8))

MTs  = cfg.get("Site-parameter", "Stations").split()
MTs = [ str(mt) for mt in MTs ]
shots = pd.read_csv(obsfile, comment='#', index_col=0)
shots = shots[~shots['flag']].reset_index(drop=True).copy()
###res
x       = shots.ResiTT
xt      = shots.ST
cvmin   = -80
cvmax   = -40
for i in range(len(xt)-1):
    if xt[i+1]-xt[i] <= 0:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
    elif xt[i+1]-xt[i] >= 3600:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
####
x_latent = np.linspace(min(xt), max(xt), 100000)
for method_name, method in [ip9]: #ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9]:
        fitted_curve = method(xt, x)
        xt = x_latent
        x  = fitted_curve(x_latent)
####
N = 2**13 #4096
hammingWindow = np.hamming(N)
samplingrate = (max(xt)-min(xt))/100000
cmap=plt.get_cmap('jet')
pxx, freqs, bins, im = axes[0,0].specgram(x, NFFT=N, Fs=1/samplingrate, noverlap=N/2, cmap=cmap, vmin=cvmin, vmax=cvmax, window=hammingWindow)
axes[0,0].set_yscale('log')
axes[0,0].set_ylim(2/N,2**-4)
axes[0,0].set_yticks([0.0002,0.0005,0.001,0.002,0.005,0.01,0.02])#]np.linspace(0.001, 0.01, 2))
axes[0,0].set_yticklabels(["5000", "2000", "1000", "500", "200", "100", "50"])
#axes[0,0].set_yticklabels(["$5*10^{3}$", "$2*10^{3}$", "$10^{3}$", "$5*10^{2}$", "$2*10^{2}$", "$10^{2}$", "$5*10$"])
axes[0,0].get_xaxis().set_ticks([])
axes[0,0].text(0.0, 0.1, "ResiTT", size = 10, color = "red")
###dV#
x       = shots.dV
xt      = shots.ST
cvmin   = -60
cvmax   =  80
for i in range(len(xt)-1):
    if xt[i+1]-xt[i] <= 0:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
    elif xt[i+1]-xt[i] >= 3600:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
k = 20
while k < len(xt)-1:
    if np.abs(x[k-19:k-1].mean()-x[k]) > 0.2:
        x[k:] = x[k:] + x[k-1] - x[k]
        k = k + 20
    else:
        k = k + 1
####
x_latent = np.linspace(min(xt), max(xt), 100000)
for method_name, method in [ip9]: #ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9]:
        fitted_curve = method(xt, x)
        xt = x_latent
        x  = fitted_curve(x_latent)
####
pxx, freqs, bins, im = axes[0,1].specgram(x, NFFT=N, Fs=1/samplingrate, noverlap=N/2, cmap=cmap, vmin=cvmin, vmax=cvmax, window=hammingWindow)
axes[0,1].set_yscale('log')
axes[0,1].set_ylim(2/N,2**-4)
axes[0,1].get_xaxis().set_ticks([])
axes[0,1].get_yaxis().set_ticks([])
axes[0,1].text(0.0, 0.1, "dV", size = 10, color = "red")
###roll
x       = shots.roll0
xt      = shots.ST
cvmin   =   0
cvmax   =  60
for i in range(len(xt)-1):
    if xt[i+1]-xt[i] <= 0:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
    elif xt[i+1]-xt[i] >= 3600:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
####
x_latent = np.linspace(min(xt), max(xt), 100000)
for method_name, method in [ip9]: #ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9]:
        fitted_curve = method(xt, x)
        xt = x_latent
        x  = fitted_curve(x_latent)
####
pxx, freqs, bins, im = axes[0,2].specgram(x, NFFT=N, Fs=1/samplingrate, noverlap=N/2, cmap=cmap, vmin=cvmin, vmax=cvmax, window=hammingWindow)
axes[0,2].set_yscale('log')
axes[0,2].set_ylim(2/N,2**-4)
axes[0,2].get_xaxis().set_ticks([])
axes[0,2].get_yaxis().set_ticks([])
axes[0,2].text(0.0, 0.1, "roll", size = 10, color = "red")
###pitch
x       = shots.pitch0
xt      = shots.ST
cvmin   =   0
cvmax   =  60
for i in range(len(xt)-1):
    if xt[i+1]-xt[i] <= 0:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
    elif xt[i+1]-xt[i] >= 3600:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
####
x_latent = np.linspace(min(xt), max(xt), 100000)
for method_name, method in [ip9]: #ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9]:
        fitted_curve = method(xt, x)
        xt = x_latent
        x  = fitted_curve(x_latent)
####
pxx, freqs, bins, im = axes[0,3].specgram(x, NFFT=N, Fs=1/samplingrate, noverlap=N/2, cmap=cmap, vmin=cvmin, vmax=cvmax, window=hammingWindow)
axes[0,3].set_yscale('log')
axes[0,3].set_ylim(2/N,2**-4)
axes[0,3].get_xaxis().set_ticks([])
axes[0,3].get_yaxis().set_ticks([])
axes[0,3].text(0.0, 0.1, "pitch", size = 10, color = "red")
###gradV1e
x       = shots.gradV1e
xt      = shots.ST
cvmin   = -80
cvmax   =  40
for i in range(len(xt)-1):
    if xt[i+1]-xt[i] <= 0:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
    elif xt[i+1]-xt[i] >= 3600:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
k = 20
while k < len(xt)-1:
    if np.abs(x[k-19:k-1].mean()-x[k]) > 0.02:
        x[k:] = x[k:] + x[k-1] - x[k]
        k = k + 20
    else:
        k = k + 1
####
x_latent = np.linspace(min(xt), max(xt), 100000)
for method_name, method in [ip9]: #ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9]:
        fitted_curve = method(xt, x)
        xt = x_latent
        x  = fitted_curve(x_latent)
####
pxx, freqs, bins, im = axes[1,0].specgram(x, NFFT=N, Fs=1/samplingrate, noverlap=N/2, cmap=cmap, vmin=cvmin, vmax=cvmax, window=hammingWindow)
axes[1,0].set_yscale('log')
axes[1,0].set_ylim(2/N,2**-4)
#axes[1,0].set_ylim((2**3)/N,2**-4)
axes[1,0].set_yticks([0.0002,0.0005,0.001,0.002,0.005,0.01,0.02])#]np.linspace(0.001, 0.01, 2))
axes[1,0].set_yticklabels(["5000", "2000", "1000", "500", "200", "100", "50"])
axes[1,0].get_xaxis().set_ticks([])
axes[1,0].text(0.0, 0.1, "gradVs(east)", size = 10, color = "red")
###gradV1n
x       = shots.gradV1n
xt      = shots.ST
for i in range(len(xt)-1):
    if xt[i+1]-xt[i] <= 0:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
    elif xt[i+1]-xt[i] >= 3600:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
k = 20
while k < len(xt)-1:
    if np.abs(x[k-19:k-1].mean()-x[k]) > 0.02:
        x[k:] = x[k:] + x[k-1] - x[k]
        k = k + 20
    else:
        k = k + 1
####
x_latent = np.linspace(min(xt), max(xt), 100000)
for method_name, method in [ip9]: #ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9]:
        fitted_curve = method(xt, x)
        xt = x_latent
        x  = fitted_curve(x_latent)
####
pxx, freqs, bins, im = axes[1,1].specgram(x, NFFT=N, Fs=1/samplingrate, noverlap=N/2, cmap=cmap, vmin=cvmin, vmax=cvmax, window=hammingWindow)
axes[1,1].set_yscale('log')
axes[1,1].set_ylim(2/N,2**-4)
#axes[1,1].set_ylim((2**3)/N,2**-4)
axes[1,1].get_xaxis().set_ticks([])
axes[1,1].get_yaxis().set_ticks([])
axes[1,1].text(0.0, 0.1, "gradVs(north)", size = 10, color = "red")
###gradV2e
x       = shots.gradV2e
xt      = shots.ST
for i in range(len(xt)-1):
    if xt[i+1]-xt[i] <= 0:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
    elif xt[i+1]-xt[i] >= 3600:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
k = 20
while k < len(xt)-1:
    if np.abs(x[k-19:k-1].mean()-x[k]) > 0.01:
        x[k:] = x[k:] + x[k-1] - x[k]
        k = k + 20
    else:
        k = k + 1
####
x_latent = np.linspace(min(xt), max(xt), 100000)
for method_name, method in [ip9]: #ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9]:
        fitted_curve = method(xt, x)
        xt = x_latent
        x  = fitted_curve(x_latent)
####
pxx, freqs, bins, im = axes[1,2].specgram(x, NFFT=N, Fs=1/samplingrate, noverlap=N/2, cmap=cmap, vmin=cvmin, vmax=cvmax, window=hammingWindow)
axes[1,2].set_yscale('log')
axes[1,2].set_ylim(2/N,2**-4)
#axes[1,2].set_ylim((2**3)/N,2**-4)
axes[1,2].get_xaxis().set_ticks([])
axes[1,2].get_yaxis().set_ticks([])
axes[1,2].text(0.0, 0.1, "gradVd(east)", size = 10, color = "red")
###gradV2n
x       = shots.gradV2n
xt      = shots.ST
for i in range(len(xt)-1):
    if xt[i+1]-xt[i] <= 0:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
    elif xt[i+1]-xt[i] >= 3600:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
k = 20
while k < len(xt)-1:
    if np.abs(x[k-19:k-1].mean()-x[k]) > 0.01:
        x[k:] = x[k:] + x[k-1] - x[k]
        k = k + 20
    else:
        k = k + 1
####
x_latent = np.linspace(min(xt), max(xt), 100000)
for method_name, method in [ip9]: #ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9]:
        fitted_curve = method(xt, x)
        xt = x_latent
        x  = fitted_curve(x_latent)
####
pxx, freqs, bins, im = axes[1,3].specgram(x, NFFT=N, Fs=1/samplingrate, noverlap=N/2, cmap=cmap, vmin=cvmin, vmax=cvmax, window=hammingWindow)
axes[1,3].set_yscale('log')
axes[1,3].set_ylim(2/N,2**-4)
#axes[1,3].set_ylim((2**3)/N,2**-4)
axes[1,3].get_xaxis().set_ticks([])
axes[1,3].get_yaxis().set_ticks([])
axes[1,3].text(0.0, 0.1, "gradVd(north)", size = 10, color = "red")
###i
len_mts = min(len(MTs),8)
for t in range(len_mts):
  tmps    = shots[ (shots['MT'] == MTs[t])  ].reset_index(drop=True).copy()
  x       = tmps.ResiTT
  xt      = tmps.ST
  cvmin   = -80
  cvmax   = -40
  for i in range(len(xt)-1):
    if xt[i+1]-xt[i] <= 0:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
    elif xt[i+1]-xt[i] >= 3600:
        xt[i+2:] = xt[i+2:] + xt[i] - xt[i+1] + 10.
        xt[i+1] = xt[i] + 10.
####
  x_latent = np.linspace(min(xt), max(xt), 100000)
  for method_name, method in [ip9]: #ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9]:
        fitted_curve = method(xt, x)
        xt = x_latent
        x  = fitted_curve(x_latent)
####
  pxx, freqs, bins, im = axes[2+(t//4),t%4].specgram(x, NFFT=N, Fs=1/samplingrate, noverlap=N/2, cmap=cmap, vmin=cvmin, vmax=cvmax, window=hammingWindow)
  axes[2+(t//4),t%4].set_yscale('log')
  axes[2+(t//4),t%4].set_ylim(2/N,2**-4)
  axes[2+(t//4),t%4].set_yticks([0.0002,0.0005,0.001,0.002,0.005,0.01,0.02])#]np.linspace(0.001, 0.01, 2))
  axes[2+(t//4),t%4].set_yticklabels(["5000", "2000", "1000", "500", "200", "100", "50"])
  axes[2+(t//4),t%4].text(0.0, 0.1, MTs[t], size = 10, color = "red")
axes[2,1].get_yaxis().set_ticks([])
axes[2,2].get_yaxis().set_ticks([])
axes[2,3].get_yaxis().set_ticks([])
axes[3,1].get_yaxis().set_ticks([])
axes[3,2].get_yaxis().set_ticks([])
axes[3,3].get_yaxis().set_ticks([])

plt.savefig('figure.png')

