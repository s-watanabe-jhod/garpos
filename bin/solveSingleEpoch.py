#!/usr/bin/env python
import os
# Disable BLAS internal threading to avoid thread oversubscription when using multiprocessing.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from optparse import OptionParser
import sys

from garpos_v120.garpos_main import drive_garpos

if __name__ == '__main__':

    ######################################################################
    usa = u"Usage: %prog [options] "
    opt = OptionParser(usage=usa)
    opt.add_option( "-i", action="store", type="string",
                    default="", dest="invcfg",
                    help=u"Path to the setup file"
                    )
    opt.add_option( "-f", action="store", type="string",
                    default="", dest="cfgfile",
                    help=u'Path to the site-parameter file'
                    )
    opt.add_option( "-d", action="store", type="string",
                    default="./result/", dest="directory",
                    help=u"Set save directory"
                    )
    opt.add_option( "--suffix", action="store", type="string",
                    default="", dest="suf",
                    help=u"Set suffix for result files"
                    )
    opt.add_option( "--maxcore", action="store", type="int",
                    default=1, dest="maxcore",
                    help=u'Set maximum CPU core'
                    )
    (options, args) = opt.parse_args()
    #####################################################################

    if not os.path.isfile(options.invcfg) or options.invcfg == "":
        print("NOT FOUND (setup file) :: %s" % options.invcfg)
        sys.exit(1)
    if not os.path.isfile(options.cfgfile) or options.cfgfile == "":
        print("NOT FOUND (site parameter file) :: %s" % options.cfgfile)
        sys.exit(1)

    odir = options.directory+"/"
    mc = options.maxcore
    rf = drive_garpos(options.cfgfile, options.invcfg, odir, options.suf, mc)

    exit()
